
import json
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\DISCERN\ml\data\raw\gc10-det")
IMG_DIR = ROOT / "ds" / "img"
ANN_DIR = ROOT / "ds" / "ann"

expected_classes = {
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist folding",
    "water_spot",
    "welding_line",
}

class_counts = Counter()
empty_files = []
invalid_boxes = []
unknown_classes = []
invalid_sizes = []

for ann_file in sorted(ANN_DIR.glob("*.json")):
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    size = data.get("size", {})
    width = size.get("width")
    height = size.get("height")

    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or width <= 0 or height <= 0:
        invalid_sizes.append(ann_file.name)
        continue

    objects = data.get("objects", [])

    if not objects:
        empty_files.append(ann_file.name)
        continue

    for index, obj in enumerate(objects):
        class_name = obj.get("classTitle", "UNKNOWN")
        class_counts[class_name] += 1

        if class_name not in expected_classes:
            unknown_classes.append((ann_file.name, class_name))

        points = obj.get("points", {}).get("exterior", [])

        if len(points) != 2:
            invalid_boxes.append((ann_file.name, index, "Invalid point count"))
            continue

        try:
            x1, y1 = points[0]
            x2, y2 = points[1]

            valid = (
                0 <= x1 < x2 <= width
                and 0 <= y1 < y2 <= height
            )

            if not valid:
                invalid_boxes.append(
                    (ann_file.name, index, points, width, height)
                )

        except (TypeError, ValueError):
            invalid_boxes.append((ann_file.name, index, "Invalid coordinates"))

print("=" * 55)
print("DISCERN - BOUNDING BOX VALIDATION")
print("=" * 55)

print("\nImages:", len(list(IMG_DIR.glob("*.jpg"))))
print("Annotations:", len(list(ANN_DIR.glob("*.json"))))
print("Total objects:", sum(class_counts.values()))

print("\n--- CLASS COUNTS ---")
for name, count in sorted(class_counts.items()):
    print(f"{name}: {count}")

print("\n--- VALIDATION RESULTS ---")
print("Empty annotations:", len(empty_files))
print("Invalid image sizes:", len(invalid_sizes))
print("Invalid bounding boxes:", len(invalid_boxes))
print("Unknown classes:", len(unknown_classes))

print("\n--- EMPTY ANNOTATION FILES ---")
for name in empty_files:
    print(name)

if invalid_sizes:
    print("\n--- INVALID IMAGE SIZES ---")
    for item in invalid_sizes[:20]:
        print(item)

if invalid_boxes:
    print("\n--- INVALID BOX EXAMPLES ---")
    for item in invalid_boxes[:20]:
        print(item)

if unknown_classes:
    print("\n--- UNKNOWN CLASS EXAMPLES ---")
    for item in unknown_classes[:20]:
        print(item)

print("\nINSPECTION COMPLETED")
