from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset_yolo" / "powerline.yaml"

    print("Using data:", data_yaml)

    # Новая модель: YOLOv8L
    model = YOLO("yolov8l.pt")   # или "yolov8x.pt", если GPU тянет

    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=640,
        batch=6,
        device=0,
        patience=30,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01
    )