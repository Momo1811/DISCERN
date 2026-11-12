
import json
from pathlib import Path
from collections import Counter

# Dataset paths
ROOT = Path(r"D:\DISCERN\ml\data\raw\gc10-det")
IMG_DIR = ROOT / "ds" / "img"
ANN_DIR = ROOT / "ds" / "ann"
META_FILE = ROOT / "meta.json"

print("=" * 55)
print("DISCERN - GC10-DET DATASET INSPECTION")
print("=" * 55)

# 1. Count images and annotations
images = list(IMG_DIR.glob("*.jpg"))
annotations = list(ANN_DIR.glob("*.json"))

print(f"\nImages: {len(images)}")
print(f"Annotation files: {len(annotations)}")

# 2. Check image-annotation matching
image_names = {p.name for p in images}
annotation_image_names = {
    p.name.removesuffix(".json") for p in annotations
}

missing_annotations = image_names - annotation_image_names
missing_images = annotation_image_names - image_names

print(f"\nImages without annotations: {len(missing_annotations)}")
print(f"Annotations without images: {len(missing_images)}")

# 3. Inspect metadata
print("\n--- METADATA ---")

with META_FILE.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Metadata keys:", list(metadata.keys()))

classes = metadata.get("classes", [])

print(f"\nNumber of classes in metadata: {len(classes)}")

for cls in classes:
    print(
        "Class:",
        cls.get("title"),
        "| Shape:",
        cls.get("shape")
    )

# 4. Count annotated defect classes
print("\n--- ANNOTATION CLASS COUNTS ---")

class_counts = Counter()
empty_annotations = 0
invalid_annotations = []

for ann_file in annotations:
    try:
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])

        if not objects:
            empty_annotations += 1

        for obj in objects:
            class_name = obj.get("classTitle", "UNKNOWN")
            class_counts[class_name] += 1

    except Exception as error:
        invalid_annotations.append((ann_file.name, str(error)))

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nTotal annotated objects:", sum(class_counts.values()))
print("Annotations with no objects:", empty_annotations)
print("Invalid annotation files:", len(invalid_annotations))

# 5. Show one annotation example
print("\n--- SAMPLE ANNOTATION ---")

if annotations:
    with annotations[0].open("r", encoding="utf-8") as f:
        sample = json.load(f)

    print("File:", annotations[0].name)
    print("Top-level keys:", list(sample.keys()))
    print("Image metadata:", sample.get("size"))
    print("First object:")

    objects = sample.get("objects", [])
    if objects:
        print(json.dumps(objects[0], indent=2))
    else:
        print("No objects in this annotation.")

print("\n" + "=" * 55)
print("INSPECTION COMPLETED")
print("=" * 55)
