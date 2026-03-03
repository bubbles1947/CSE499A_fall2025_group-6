import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("Exporting Vision Encoder to ONNX...")
print("=" * 50)

# Load model from local cache
print("\n[1/4] Loading model from cache...")
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    cache_dir="./models"
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./models"
)
model.eval()
print("Model loaded from cache!")

# Extract only the vision encoder
print("\n[2/4] Extracting vision encoder...")
vision_model = model.vision_model
vision_model = vision_model.to("cuda").half()
vision_model.eval()
print("Vision encoder extracted!")

# Create dummy input for export
print("\n[3/4] Preparing dummy input...")
dummy_input = torch.randn(1, 3, 224, 224).to("cuda").half()
print("Dummy input ready!")

# Export to ONNX
print("\n[4/4] Exporting to ONNX (this may take 2-5 minutes)...")
import os
os.makedirs("./onnx_models", exist_ok=True)

try:
    torch.onnx.export(
        vision_model,
        dummy_input,
        "./onnx_models/blip2_vision_encoder.onnx",
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"}
        },
        do_constant_folding=True
    )
    print("ONNX export successful!")
except Exception as e:
    print(f"Export error: {e}")

# Verify the exported file
import os
if os.path.exists("./onnx_models/blip2_vision_encoder.onnx"):
    size_mb = os.path.getsize("./onnx_models/blip2_vision_encoder.onnx") / (1024 * 1024)
    print(f"\nExported file size: {size_mb:.1f} MB")
    print("Vision Encoder ONNX export complete!")
else:
    print("Export failed - file not found!")

print("\n" + "=" * 50)
print("Done! Check onnx_models folder.")
print("=" * 50)
