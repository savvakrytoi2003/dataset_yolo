import json
from pathlib import Path

# пути к файлам
IN_PATH = Path(r"C:\Users\SAVVA\Desktop\hakaton_v2\dataset\coco_categories.json")
OUT_PATH = Path(r"C:\Users\SAVVA\Desktop\hakaton_v2\dataset\coco_categories_filtered.json")

# какие категории удаляем
REMOVE_CATS = {2220001, 2270001}  # nest, safety_sign+

with IN_PATH.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

print("Ключи в файле:", list(cfg.keys()))   # должно быть ['categories']
print("Категорий до:", len(cfg["categories"]))

cfg["categories"] = [
    c for c in cfg["categories"]
    if c["id"] not in REMOVE_CATS
]

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("Категорий после:", len(cfg["categories"]))
print("Сохранено в:", OUT_PATH)
