import cv2
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]          # routers -> computer_vision -> hakaton_v2
model_path  = PROJECT_ROOT / "best.onnx"
img_path    = PROJECT_ROOT / "img.png"
out_path    = PROJECT_ROOT / "result.jpg"

INPUT_W = 640
INPUT_H = 640
CONF_THRES = 0.4
NMS_THRES  = 0.5

print("MODEL_PATH:", model_path, "exists:", model_path.exists())
print("IMG_PATH:", img_path, "exists:", img_path.exists())

# --- загрузка модели ---
net = cv2.dnn.readNetFromONNX(str(model_path))

# --- картинка ---
img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError("cv2.imread вернул None, проверь путь к img.jpg")

orig_h, orig_w = img.shape[:2]

# --- препроцесс ---
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,
    size=(INPUT_W, INPUT_H),
    swapRB=True,
    crop=False,
)
net.setInput(blob)

# --- прямой проход ---
out = net.forward()
print("raw out shape:", out.shape)   # ожидаемо что-то типа (1, C, 8400)

# YOLOv8 ONNX: (1, C, N) -> (N, C)
out = out[0]                # (C, N)
out = out.transpose(1, 0)   # (N, C)

num_preds, ch = out.shape
num_classes = ch - 4
print("num_preds:", num_preds, "channels:", ch, "num_classes:", num_classes)

# первые 4 значения — bbox (cx, cy, w, h)
boxes_xywh = out[:, :4]
cls_scores = out[:, 4:]     # уже после sigmoid, можно использовать как есть

# --- вытаскиваем лучшую вероятность и класс ---
if num_classes == 1:
    scores = cls_scores[:, 0]
    class_ids = np.zeros_like(scores, dtype=int)
else:
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(num_preds), class_ids]

# фильтр по порогу
mask = scores > CONF_THRES
boxes_xywh = boxes_xywh[mask]
scores     = scores[mask]
class_ids  = class_ids[mask]

print("after conf filter:", len(boxes_xywh))

if len(boxes_xywh) == 0:
    print("⚠️ Модель ничего уверенно не нашла на этой картинке.")
else:
    # если увидишь, что координаты ~0..1, раскомментируй этот блок:
    # boxes_xywh[:, 0] *= INPUT_W
    # boxes_xywh[:, 1] *= INPUT_H
    # boxes_xywh[:, 2] *= INPUT_W
    # boxes_xywh[:, 3] *= INPUT_H

    # cx,cy,w,h -> x,y,w,h (левый верхний)
    boxes = []
    for cx, cy, w, h in boxes_xywh:
        x1 = cx - w / 2
        y1 = cy - h / 2
        boxes.append([int(x1), int(y1), int(w), int(h)])

    # масштабирование с 640x640 обратно в исходный размер
    scale_x = orig_w / INPUT_W
    scale_y = orig_h / INPUT_H

    scaled_boxes = []
    for x, y, w, h in boxes:
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        scaled_boxes.append([x, y, w, h])

    # NMS
    indices = cv2.dnn.NMSBoxes(scaled_boxes, scores.tolist(), CONF_THRES, NMS_THRES)

    if len(indices) == 0:
        print("⚠️ После NMS боксов нет (всё задавили друг друга).")
    else:
        print("boxes after NMS:", len(indices))
        for i in indices.flatten():
            x, y, w, h = scaled_boxes[i]
            conf = float(scores[i])
            cls  = int(class_ids[i])

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{cls}:{conf:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

# --- сохраняем картинку ---
cv2.imwrite(str(out_path), img)
print("✅ Результат сохранён в", out_path)
