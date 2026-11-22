#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "arena_allocator.hpp"
#include "ring_buffer.hpp"

struct FramePacket { int id; cv::Mat frame; };
struct Detection { std::string label; float conf; cv::Rect box; };

void pin_thread(int core) {
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

double get_rss_mb() {
    struct rusage usage; getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

void preprocess_into_arena(const cv::Mat& img, float* dst) {
    const int H = 640, W = 640, plane = H * W;
    float *r = dst, *g = dst + plane, *b = dst + 2 * plane;
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        int off = y * W;
        for (int x = 0; x < W; ++x) {
            b[off + x] = row[x][0] * (1.0f / 255.0f);
            g[off + x] = row[x][1] * (1.0f / 255.0f);
            r[off + x] = row[x][2] * (1.0f / 255.0f);
        }
    }
}

std::string class_name(int id) {
    switch (id) {
        case 0: return "PERSON"; case 2: return "CAR"; case 5: return "BUS";
        case 7: return "TRUCK"; case 16: return "DOG"; case 17: return "HORSE";
        case 18: return "SHEEP"; case 19: return "CATTLE"; default: return "OBSTACLE";
    }
}

std::vector<Detection> decode_hazards(const float* out, float thresh = 0.45f) {
    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> cls_ids;
    const int anchors = 8400;
    const int targets[] = {0, 2, 5, 7, 16, 17, 18, 19};
    for (int i = 0; i < anchors; ++i) {
        float best = 0.0f; int best_c = -1;
        for (int c : targets) {
            float s = out[(4 + c) * anchors + i];
            if (s > best) { best = s; best_c = c; }
        }
        if (best >= thresh) {
            float cx = out[0 * anchors + i], cy = out[1 * anchors + i];
            float w  = out[2 * anchors + i], h  = out[3 * anchors + i];
            boxes.emplace_back(int(cx - w/2), int(cy - h/2), int(w), int(h));
            scores.push_back(best); cls_ids.push_back(best_c);
        }
    }
    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, thresh, 0.45f, idx);
    std::vector<Detection> dets;
    for (int k : idx) dets.push_back({class_name(cls_ids[k]), scores[k], boxes[k]});
    return dets;
}

int main(int argc, char** argv) {
    LockFreeRingBuffer<FramePacket, 16> ring;
    std::atomic<bool> done{false};
    std::string video_path = (argc > 1) ? argv[1] : "../data/test_track.mp4";

    std::thread producer([&]() {
        pin_thread(1);
        cv::VideoCapture cap(video_path);
        bool ok = cap.isOpened();
        for (int i = 0; i < 1500; ++i) {
            cv::Mat f;
            if (ok) {
                if (!cap.read(f) || f.empty()) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); cap.read(f); } if (f.empty()) continue;
            } else {
                f = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(45, 55, 50));
                cv::putText(f, "SIMULATED CORRIDOR STREAM", {60, 360}, cv::FONT_HERSHEY_SIMPLEX, 1.2, {200,220,200}, 2);
            }
            while (!ring.push({i, std::move(f)})) { std::this_thread::yield(); }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        done.store(true);
    });

    // Allow ONNX intra-op pool to scale across physical cores
    CacheAlignedArena arena(32 * 1024 * 1024);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RailGuardARM64");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, "../models/yolov8n_fp32.onnx", opts);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, 640, 640};
    const char *in_names[] = {"images"}, *out_names[] = {"output0"};
    std::vector<double> latencies;

    while (!done.load()) {
        FramePacket pkt;
        if (!ring.pop(pkt)) { std::this_thread::yield(); continue; }

        arena.reset();
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Rect roi(0, pkt.frame.rows / 4, pkt.frame.cols, (pkt.frame.rows * 3) / 4);
        cv::Mat resized;
        if (pkt.frame.empty()) continue;
        cv::resize(pkt.frame(roi), resized, cv::Size(640, 640));

        float* buf = static_cast<float*>(arena.allocate(3 * 640 * 640 * sizeof(float)));
        preprocess_into_arena(resized, buf);

        Ort::Value tensor = Ort::Value::CreateTensor<float>(mem_info, buf, 3 * 640 * 640, shape.data(), 4);
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto dets = decode_hazards(outs[0].GetTensorData<float>(), 0.45f);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (pkt.id >= 40) latencies.push_back(ms);

        if (pkt.id % 5 == 0 && !latencies.empty()) {
            for (const auto& d : dets) {
                cv::rectangle(resized, d.box, cv::Scalar(0, 0, 255), 2);
                cv::putText(resized, d.label, {d.box.x, std::max(15, d.box.y - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
            }
            cv::imwrite("../../dashboard/live_frame.jpg", resized);

            std::vector<double> s = latencies;
            std::sort(s.begin(), s.end());
            double p50 = s[s.size() * 0.50], p99 = s[s.size() * 0.99];
            double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
            double fps = 1000.0 / mean;
            double cost_1m = (0.0725 / (fps * 3600.0)) * 1000000.0;

            std::ofstream jf("../../dashboard/telemetry_feed.json");
            jf << "{\"frame\":" << pkt.id << ",\"p50_ms\":" << p50 << ",\"p99_ms\":" << p99
               << ",\"fps\":" << fps << ",\"peak_ram_mb\":" << get_rss_mb()
               << ",\"cost_per_1m_usd\":" << cost_1m << ",\"hazards\":" << dets.size() << "}";
        }
    }
    producer.join();
    return 0;
}
