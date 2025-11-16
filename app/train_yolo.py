from ultralytics import YOLO
import os
from pathlib import Path


if __name__ == "__main__":
    # корень проекта: .../hakaton_v2
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset_yolo" / "powerline.yaml"

    print("Using data:", data_yaml)

    # загружаем предобученную маленькую модель
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=8,
        device=0
    )