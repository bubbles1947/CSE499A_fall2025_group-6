import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("Testing Full Caption Pipeline...")
print("=" * 50)

# Load from cache
print("\n[1/3] Loading model from cache...")
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
print("Model loaded!")

# Test with multiple images
print("\n[2/3] Running inference on test images...")

test_images = [
    "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
]

for i, url in enumerate(test_images):
    try:
        image = Image.open(
            requests.get(url, stream=True).raw
        ).convert("RGB")
        inputs = processor(
            image,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50
            )
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        print(f"\nImage {i+1}: {url.split('/')[-1]}")
        print(f"Caption : {caption}")
    except Exception as e:
        print(f"Image {i+1} skipped: {e}")

# Test with a local image if exists
print("\n[3/3] Checking for local test image...")
import os
if os.path.exists("./test_image.jpg"):
    image = Image.open("./test_image.jpg").convert("RGB")
    inputs = processor(
        image,
        return_tensors="pt"
    ).to("cuda", torch.float16)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50
        )
    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()
    print(f"Local image caption: {caption}")
else:
    print("No local test image found.")
    print("Tip: Put any image as 'test_image.jpg' in this folder to test!")

print("\n" + "=" * 50)
print("Pipeline test complete!")
print("=" * 50)