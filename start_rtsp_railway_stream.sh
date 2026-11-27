#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
U="https://github.com"
[ ! -f mediamtx ] && { [ "$(uname -m)" = "aarch64" ] && wget -q "$U/mediamtx_v1.9.3_linux_arm64v8.tar.gz" -O m.tgz || wget -q "$U/mediamtx_v1.9.3_linux_amd64.tar.gz" -O m.tgz; tar -xzf m.tgz mediamtx && rm m.tgz; }
pkill -f mediamtx || true; pkill -f "ffmpeg.*8554" || true
./mediamtx > /tmp/mtx.log 2>&1 & sleep 2
ffmpeg -re -stream_loop -1 -i data/scenario_level_crossing.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/crossing_cam_01 > /tmp/ff.log 2>&1 & sleep 2
cd cpp_engine/build && LD_LIBRARY_PATH=../../third_party/onnxruntime/lib ./railguard_engine rtsp://localhost:8554/crossing_cam_01
