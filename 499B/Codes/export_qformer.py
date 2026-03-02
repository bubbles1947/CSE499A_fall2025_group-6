import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("Exporting Q-Former to ONNX...")
print("=" * 50)

# Load model from cache
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

# Extract Q-Former
print("\n[2/4] Extracting Q-Former...")

class QFormerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.qformer = model.qformer
        self.query_tokens = model.query_tokens
        self.language_projection = model.language_projection

    def forward(self, image_features):
        batch_size = image_features.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
            return_dict=True
        )
        projected = self.language_projection(
            query_output.last_hidden_state
        )
        return projected

qformer_wrapper = QFormerWrapper(model)
qformer_wrapper = qformer_wrapper.to("cuda").half()
qformer_wrapper.eval()
print("Q-Former extracted!")

# Create dummy input
print("\n[3/4] Preparing dummy input...")
dummy_image_features = torch.randn(1, 257, 1408).to("cuda").half()
print("Dummy input ready!")

# Export to ONNX
print("\n[4/4] Exporting Q-Former to ONNX...")
os.makedirs("./onnx_models", exist_ok=True)

try:
    with torch.no_grad():
        torch.onnx.export(
            qformer_wrapper,
            dummy_image_features,
            "./onnx_models/blip2_qformer.onnx",
            opset_version=14,
            input_names=["image_features"],
            output_names=["language_features"],
            dynamic_axes={
                "image_features": {0: "batch_size"},
                "language_features": {0: "batch_size"}
            },
            do_constant_folding=True
        )
    print("Q-Former ONNX export successful!")
except Exception as e:
    print(f"Export error: {e}")

# Check file
if os.path.exists("./onnx_models/blip2_qformer.onnx"):
    size_mb = os.path.getsize(
        "./onnx_models/blip2_qformer.onnx"
    ) / (1024 * 1024)
    print(f"\nQ-Former file size: {size_mb:.1f} MB")
    print("Q-Former export complete!")
else:
    print("Export failed - file not found!")

print("\n" + "=" * 50)
print("Done! Check onnx_models folder.")
print("=" * 50)
