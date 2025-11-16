import os
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import torch

def main():
    # корень проекта: .../hakaton_v2
    project_root = Path(__file__).resolve().parents[1]

    coco_ann_path = project_root / "dataset" / "annotation_data.json"
    images_root = project_root / "dataset" / "images"
    out_root = project_root / "dataset_yolo"

    print("project_root:", project_root)
    print("coco_ann_path:", coco_ann_path)
    print("images_root:", images_root)
    print("out_root:", out_root)

    out_images = out_root / "images"
    out_labels = out_root / "labels"

    # создаём папки
    for split in ["train", "val", "test"]:
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    # грузим COCO
    with open(coco_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # cat_id -> yolo_class_idx (0..N-1)
    sorted_cat_ids = sorted(c["id"] for c in categories)
    catid2idx = {cat_id: i for i, cat_id in enumerate(sorted_cat_ids)}
    catid2name = {c["id"]: c["name"] for c in categories}
    names_ordered = [catid2name[cid] for cid in sorted_cat_ids]

    print("classes:")
    for i, name in enumerate(names_ordered):
        print(f"{i}: {name}")

    # image_id -> список аннотаций
    imgid2anns = defaultdict(list)
    for a in annotations:
        imgid2anns[a["image_id"]].append(a)

    # делим на train/val/test
    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    def process_split(split_name, split_images):
        print(f"Processing {split_name}, images: {len(split_images)}")
        for img in split_images:
            file_name = img["file_name"]  # относительный путь внутри dataset/images
            src_img = images_root / file_name

            # куда скопировать изображение
            dst_img = out_images / split_name / file_name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)

            w, h = img["width"], img["height"]

            # путь для txt-разметки
            label_rel = Path(file_name).with_suffix(".txt")
            label_path = out_labels / split_name / label_rel
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for ann in imgid2anns[img["id"]]:
                cat_id = ann["category_id"]
                cls = catid2idx[cat_id]

                # COCO bbox = [x_min, y_min, w, h] (в пикселях)
                x, y, bw, bh = ann["bbox"]
                x_c = x + bw / 2
                y_c = y + bh / 2

                # нормализация
                x_c /= w
                y_c /= h
                bw /= w
                bh /= h

                lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    process_split("train", train_imgs)
    process_split("val", val_imgs)
    process_split("test", test_imgs)

    # пишем YAML для YOLO
    yaml_path = out_root / "powerline.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {str(out_root).replace('\\', '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for i, name in enumerate(names_ordered):
            f.write(f"  {i}: {name}\n")

    print("Done. YAML:", yaml_path)


if __name__ == "__main__":
    print(torch.cuda.is_available())