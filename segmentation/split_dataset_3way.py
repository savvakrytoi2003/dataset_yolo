import os
import shutil
import random
from pathlib import Path

dataset_dir = Path("dataset_insulator_seg")

# теперь ИСХОДНАЯ папка — images/train
images_dir = dataset_dir / "images" / "train"
labels_dir = dataset_dir / "labels" / "train"

if not images_dir.exists():
    raise RuntimeError("Папка dataset_insulator_seg/images/train не найдена")

if not labels_dir.exists():
    raise RuntimeError("Папка dataset_insulator_seg/labels/train не найдена")

# Создаём папки val/test (train уже существует)
for split in ["val", "test"]:
    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# собираем список изображений
images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))

print(f"Найдено изображений в train/: {len(images)}")

random.shuffle(images)

# Разбиение
n = len(images)
train_size = int(n * 0.7)
val_size = int(n * 0.15)
test_size = n - train_size - val_size

new_train_imgs = images[:train_size]
val_imgs = images[train_size: train_size + val_size]
test_imgs = images[train_size + val_size:]

def move_split(img_list, split_name):
    for img_path in img_list:
        label_path = labels_dir / f"{img_path.stem}.txt"

        # перемещаем в split
        shutil.move(str(img_path), dataset_dir / "images" / split_name / img_path.name)

        if label_path.exists():
            shutil.move(str(label_path), dataset_dir / "labels" / split_name / label_path.name)
        else:
            print(f"[WARN] Нет label для {img_path.name}")

# Переносим val/test
move_split(val_imgs, "val")
move_split(test_imgs, "test")

# train остаётся на месте, но нужно перенести только лишнее
# уберём остатки label из train, которых нет
existing_train = set([p.stem for p in new_train_imgs])
for lbl in labels_dir.glob("*.txt"):
    if lbl.stem not in existing_train:
        shutil.move(str(lbl), dataset_dir / "labels" / "train" / lbl.name)

print("Разделение train/val/test завершено!")

# создаём data.yaml
yaml_path = dataset_dir / "data.yaml"
yaml_text = f"""
path: {dataset_dir}
train: images/train
val: images/val
test: images/test
names:
  0: insulator
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_text.strip())

print(f"Создан корректный data.yaml → {yaml_path}")
