from ultralytics import YOLO
from pathlib import Path


if __name__ == "__main__":
    # корень проекта, как в train_yolo.py
    project_root = Path(__file__).resolve().parents[0]  # если скрипт лежит в корне проекта
    # если хочешь точно так же, как в старом файле (родитель выше), то:
    # project_root = Path(__file__).resolve().parents[1]

    # путь до нашего сегментационного датасета
    data_yaml = project_root / "dataset_insulator_seg" / "data.yaml"

    # ─────────────────────────────────
    # ВАРИАНТ 1. Обучение от предобученных весов COCO (рекомендуется)
    # ─────────────────────────────────
    model = YOLO("yolov8n-seg.pt")  # можешь заменить на yolov8s-seg.pt / m / l

    # # ─────────────────────────────────
    # # ВАРИАНТ 2. Полностью новая модель (без предобученных весов)
    # # ─────────────────────────────────
    # model = YOLO("yolov8n-seg.yaml")  # рандомная инициализация

    model.train(
        data=str(data_yaml),
        epochs=100,       # можешь поставить 120 как в старом скрипте
        imgsz=640,
        batch=4,
        device=0,         # 0 = первая GPU; "cpu" если без GPU
        patience=30,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,

        # те же аугментации, что ты юзал раньше для bbox,
        # для сегментации тоже норм
        hsv_h=0.015,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.1,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.1,
    )
