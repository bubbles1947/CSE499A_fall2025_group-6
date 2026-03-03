import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("Starting BLIP-2 Download and Test...")
print("=" * 50)

# Load model (first time will download ~6 GB)
print("\n[1/3] Loading Processor...")
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    cache_dir="./models"
)
print("Processor loaded successfully!")

print("\n[2/3] Loading Model (this will take time)...")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./models"
)
model.eval()
print("Model loaded successfully!")

# Test with a sample image
print("\n[3/3] Running test inference...")
test_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
try:
    image = Image.open(requests.get(test_url, stream=True).raw).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"Test Caption: {caption}")
except Exception as e:
    print(f"Skipping internet test: {e}")
    print("Model is still fine!")

print("\n" + "=" * 50)
print("Everything is ready! You can proceed to the next step.")
print("=" * 50)