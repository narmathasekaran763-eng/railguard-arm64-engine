import os
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO("yolov8n.pt")
fp32_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
os.rename(fp32_path, "models/yolov8n_fp32.onnx")

quantize_dynamic(
    model_input="models/yolov8n_fp32.onnx",
    model_output="models/yolov8n_int8.onnx",
    weight_type=QuantType.QUInt8
)
print("Models exported successfully.")
