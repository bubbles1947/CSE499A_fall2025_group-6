import os
from pathlib import Path

DATASET_PATH = "./plantvillage_dataset"

print("=" * 80)
print("DATASET DIAGNOSTIC")
print("=" * 80)

# Check if folder exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset folder does not exist!")
    print(f"   Path: {DATASET_PATH}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"\n   Please create folder: {os.path.abspath(DATASET_PATH)}")
    exit()

print(f"\n✅ Dataset folder found at: {DATASET_PATH}")

# List all items in dataset folder
items = os.listdir(DATASET_PATH)
print(f"\n📁 Contents of {DATASET_PATH}:")
for item in sorted(items)[:15]:  # Show first 15
    path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(path):
        files = os.listdir(path)
        print(f"   📂 {item}/ ({len(files)} files)")
    else:
        print(f"   📄 {item}")

# Count disease classes (directories)
classes = [d for d in os.listdir(DATASET_PATH) 
           if os.path.isdir(os.path.join(DATASET_PATH, d))]

print(f"\n📊 Statistics:")
print(f"   Total folders: {len(items)}")
print(f"   Disease classes: {len(classes)}")

if len(classes) == 0:
    print(f"\n❌ ERROR: No disease class folders found!")
    print(f"   Dataset folder seems empty or wrongly extracted")
    exit()

# Check each class for images
print(f"\n🔍 Checking images in each class:")
total_images = 0
classes_with_images = 0

for i, cls in enumerate(sorted(classes)):
    cls_path = os.path.join(DATASET_PATH, cls)
    
    # List all files
    all_files = os.listdir(cls_path)
    
    # Filter for image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = [f for f in all_files if f.lower().endswith(image_extensions)]
    
    total_images += len(images)
    if len(images) > 0:
        classes_with_images += 1
    
    # Show first few and last few
    if i < 5 or i >= len(classes) - 5:
        status = "✅" if len(images) > 0 else "❌"
        print(f"   {status} {cls}: {len(images)} images")
    elif i == 5 and len(classes) > 10:
        print(f"   ... ({len(classes) - 10} more classes)")

print(f"\n📈 Summary:")
print(f"   Total classes with images: {classes_with_images}/{len(classes)}")
print(f"   Total images found: {total_images}")

if total_images == 0:
    print(f"\n❌ CRITICAL ERROR: No images found in dataset!")
    print(f"\n   This could mean:")
    print(f"   1. Dataset not properly extracted")
    print(f"   2. Images have wrong file extensions")
    print(f"   3. Dataset structure is incorrect")
    print(f"\n   Expected structure:")
    print(f"   plantvillage_dataset/")
    print(f"   ├── Apple___Apple_scab/")
    print(f"   │   ├── image001.jpg")
    print(f"   │   ├── image002.jpg")
    print(f"   │   └── ...")
    print(f"   ├── Apple___Black_rot/")
    print(f"   └── ...")
else:
    print(f"\n✅ Dataset looks good! Ready for training")

print("\n" + "=" * 80)