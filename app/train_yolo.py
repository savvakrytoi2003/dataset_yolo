from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset_yolo" / "powerline.yaml"

    model = YOLO("best.pt")

    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=640,
        batch=4,
        device=0,
        patience=30,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,

        # 🔧 аугментации
        hsv_h=0.015,   # лёгкий сдвиг оттенканость
        hsv_v=0.4,     # яркость

        degrees=5.0,   # небольшие повороты (ЛЭП всё ещё стоят вертикально)
        translate=0.1, # до 10% сдвига
        scale=0.1,     # +/-10% масштаб
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,    # горизонтальный флип (для ЛЭП норм)
        mosaic=0.7,    # не максимальный, чтобы не разрывать сцены
        mixup=0.1,
        copy_paste=0.1,
    )
