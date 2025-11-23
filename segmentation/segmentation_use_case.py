import cv2
import numpy as np
from ultralytics import YOLO

# === ПУТЬ К ТВОЕЙ МОДЕЛИ ===
model = YOLO(r"C:\Users\SAVVA\Desktop\hakaton_v2\segmentation\runs\segment\train5\weights\best.pt")

# === ПУТЬ К ФОТОГРАФИИ ===
img_path = r"C:\Users\SAVVA\Desktop\hakaton_v2\test2.jpg"

# Загружаем изображение через OpenCV
img = cv2.imread(img_path)
orig = img.copy()

# Запускаем сегментацию
results = model.predict(img, device=0)  # GPU=0, CPU="cpu"

# Берём первое предсказание
res = results[0]

if res.masks is None:
    print("Масок не найдено!")
else:
    # Перебираем все найденные объекты
    for mask, box, cls in zip(res.masks.data, res.boxes.xyxy, res.boxes.cls):

        # === Маска ===
        m = mask.cpu().numpy()
        m = cv2.resize(m, (img.shape[1], img.shape[0]))
        m = (m > 0.5).astype(np.uint8)

        # Цвет маски
        color = (0, 255, 0)

        # Накладываем маску на картинку
        img[m == 1] = img[m == 1] * 0.5 + np.array(color) * 0.5

        # === Контур маски ===
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)

        # === Бокс ===
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # === Подпись ===
        class_name = "insulator"  # у тебя 1 класс
        cv2.putText(img, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Показываем результат
cv2.imshow("YOLOv8-Seg result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
