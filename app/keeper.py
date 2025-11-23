import json
import random
import shutil
from pathlib import Path


# === НАСТРОЙКИ ===

# Папка с картинками и COCO-аннотацией
IMAGES_DIR = Path(r"C:\Users\SAVVA\Desktop\efiCOCO json
COCO_JSON_PATH = IMAGES_DIR / "annotation_data.json"   # поменяй имя, если нужно

# Во сколько раз уменьшать датасет (10 -> оставляем 1/10)
REDUCE_FACTOR = 10

# Куда класть результат (относительно IMAGES_DIR)
OUTPUT_SUBDIR = "split_by_category_10x_less"

# Фиксированное зерно для воспроизводимого случайного выбора
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    if not COCO_JSON_PATH.exists():
        raise FileNotFoundError(f"Не найден COCO json: {COCO_JSON_PATH}")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с картинками: {IMAGES_DIR}")

    print(f"[INFO] Читаю COCO файл: {COCO_JSON_PATH}")

    with COCO_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    # id изображения -> имя файла
    image_id_to_name = {img["id"]: img["file_name"] for img in images}

    # id категории -> имя категории
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # image_id -> множество category_id
    img_to_cats = {}
    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_to_cats.setdefault(img_id, set()).add(cat_id)

    # category_id -> множество image_id
    cat_to_imgs = {}
    for img_id, cat_ids in img_to_cats.items():
        for cid in cat_ids:
            cat_to_imgs.setdefault(cid, set()).add(img_id)

    out_root = IMAGES_DIR / OUTPUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Пишу результат в: {out_root}")

    fraction = 1.0 / REDUCE_FACTOR

    for cat_id, img_ids in cat_to_imgs.items():
        cat_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
        safe_cat_name = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in cat_name
        )
        out_cat_dir = out_root / safe_cat_name
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        img_ids = list(img_ids)
        n_total = len(img_ids)
        if n_total == 0:
            continue

        n_keep = max(1, int(n_total * fraction))
        selected_ids = set(random.sample(img_ids, n_keep))

        print(
            f"[INFO] Категория '{cat_name}' ({cat_id}): "
            f"{n_total} изображений -> {n_keep}"
        )

        copied = 0
        for img_id in selected_ids:
            file_name = image_id_to_name.get(img_id)
            if not file_name:
                continue

            src = IMAGES_DIR / file_name
            if not src.exists():
                print(f"[WARN] Не найден файл: {src}")
                continue

            dst = out_cat_dir / src.name
            shutil.copy2(src, dst)
            copied += 1

        print(f"[INFO]   Скопировано {copied} файлов в {out_cat_dir}")

    print("[DONE] Готово.")


if __name__ == "__main__":
    main()
