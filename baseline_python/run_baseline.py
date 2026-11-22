import time, os, json, psutil, cv2
import numpy as np
import onnxruntime as ort

process = psutil.Process(os.getpid())
session = ort.InferenceSession("models/yolov8n_fp32.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture("data/test_track.mp4")
has_video = cap.isOpened()
latencies = []

for i in range(200):
    ret, frame = cap.read() if has_video else (True, np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()

    t0 = time.perf_counter()
    resized = cv2.resize(frame, (640, 640))
    blob = np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
    _ = session.run(None, {input_name: blob})
    t1 = time.perf_counter()
    if i >= 15:
        latencies.append((t1 - t0) * 1000.0)

arr = np.array(latencies)
fps = 1000.0 / float(np.mean(arr))
rss = process.memory_info().rss / (1024 * 1024)
cost_1m = (1.006 / (fps * 3600.0)) * 1_000_000.0

os.makedirs("dashboard", exist_ok=True)
with open("dashboard/baseline_metrics.json", "w") as f:
    json.dump({
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps": round(fps, 1),
        "peak_ram_mb": round(rss, 1),
        "cost_per_1m_usd": round(cost_1m, 4)
    }, f, indent=2)
print("Saved baseline_metrics.json")
