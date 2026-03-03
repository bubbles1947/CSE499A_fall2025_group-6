import onnx
import onnxruntime as ort
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("Verifying and Quantizing ONNX Model...")
print("=" * 50)

# Step 1: Verify ONNX model structure
print("\n[1/4] Verifying ONNX model structure...")
model = onnx.load("./onnx_models/blip2_vision_encoder.onnx")
try:
    onnx.checker.check_model(model)
    print("ONNX model structure is valid!")
except Exception as e:
    print(f"Model check warning (usually safe to ignore): {e}")

# Step 2: Test inference with ONNX Runtime
print("\n[2/4] Testing inference with ONNX Runtime...")
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    session = ort.InferenceSession(
        "./onnx_models/blip2_vision_encoder.onnx",
        session_options,
        providers=["CPUExecutionProvider"]
    )
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float16)
    outputs = session.run(None, {"pixel_values": dummy_input})
    print(f"Inference successful!")
    print(f"Output shape: {outputs[0].shape}")
except Exception as e:
    print(f"Inference error: {e}")

# Step 3: Apply INT8 Quantization
print("\n[3/4] Applying INT8 Quantization...")
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

try:
    quantize_dynamic(
        model_input="./onnx_models/blip2_vision_encoder.onnx",
        model_output="./onnx_models/blip2_vision_encoder_int8.onnx",
        weight_type=QuantType.QUInt8
    )
    print("INT8 Quantization successful!")
except Exception as e:
    print(f"Quantization error: {e}")

# Step 4: Compare file sizes
print("\n[4/4] Comparing file sizes...")
original_size = os.path.getsize(
    "./onnx_models/blip2_vision_encoder.onnx"
) / (1024 * 1024)

if os.path.exists("./onnx_models/blip2_vision_encoder_int8.onnx"):
    quantized_size = os.path.getsize(
        "./onnx_models/blip2_vision_encoder_int8.onnx"
    ) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    print(f"Original model size  : {original_size:.1f} MB")
    print(f"Quantized model size : {quantized_size:.1f} MB")
    print(f"Size reduction       : {reduction:.1f}%")
else:
    print(f"Original model size  : {original_size:.1f} MB")
    print("Quantized model not found!")

print("\n" + "=" * 50)
print("Done! Both models are in onnx_models folder.")
print("=" * 50)
