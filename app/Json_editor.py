import json
from pathlib import Path

coco_path = Path("instances_train.json")
out_path = Path("instances_train_filtered.json")

REMOVE_CATS = {2220001, 2270001}  # nest, safety_sign+

with coco_path.open() as f:
    coco = json.load(f)

# 1. оставляем только нужные категории
coco["categories"] = [
    c for c in coco["categories"]
    if c["id"] not in REMOVE_CATS
]

# 2. фильтруем аннотации
keep_anns = [
    ann for ann in coco["annotations"]
    if ann["category_id"] not in REMOVE_CATS
]

# 3. выкидываем картинки, у которых после фильтрации не осталось ни одной аннотации
img_has_ann = {ann["image_id"] for ann in keep_anns}
coco["images"] = [
    img for img in coco["images"]
    if img["id"] in img_has_ann
]
coco["annotations"] = keep_anns

with out_path.open("w") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print("saved to", out_path)
