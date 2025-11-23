from ultralytics import YOLO
from pathlib import Path



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset_yolo" / "powerline.yaml"

    print("Using data:", data_yaml)

    model = YOLO("last.pt")

    model.train(
        data=str(data_yaml),
        epochs=120,
        patience=30,
        resume=True
    )