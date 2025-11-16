import os
import json

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset


class PowerLineCocoDataset(Dataset):
    """
    Датасет для кейса ЦИТ (детекция виброгасителей, изоляторов, траверс и т.п.)
    по COCO-аннотациям из annotation_data.json.

    Возвращает:
        image: 3xHxW (после transform)
        target: dict, совместимый с torchvision detection:
            - boxes: [N, 4] (x, y, w, h)
            - labels: [N]
            - image_id: scalar tensor
            - area: [N]
            - iscrowd: [N]
            - (опционально) rotation: [N]
    """

    def __init__(
        self,
        images_root: str,
        ann_file: str,
        transform=None,
        only_eval_classes: bool = False,
    ):
        """
        :param images_root: путь к папке, где лежат все подпапки с картинками.
                            Пример: "/data/lep_images"
                            Тогда file_name из json подставляется так:
                            os.path.join(images_root, file_name)
        :param ann_file: путь к annotation_data.json
        :param transform: torchvision.transforms для картинки (PIL -> Tensor и т.п.)
        :param only_eval_classes: если True, оставляем только 6 классов из кейса:
            vibration_damper, festoon_insulators, traverse,
            bad_insulator, damaged_insulator, polymer_insulators
        """
        self.images_root = images_root
        self.transform = transform

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        # --- фильтр категорий (если нужно только 6 из кейса) ---
        if only_eval_classes:
            eval_names = {
                "vibration_damper",
                "festoon_insulators",
                "traverse",
                "bad_insulator",
                "damaged_insulator",
                "polymer_insulators",
            }
            self.categories = [
                c for c in self.categories if c["name"] in eval_names
            ]

        # id категории -> индекс [0..num_classes-1]
        sorted_cat_ids = sorted(c["id"] for c in self.categories)
        self.catid2idx = {cat_id: i for i, cat_id in enumerate(sorted_cat_ids)}
        self.idx2name = {
            self.catid2idx[c["id"]]: c["name"]
            for c in self.categories
            if c["id"] in self.catid2idx
        }

        # image_id -> список аннотаций (с учётом фильтра по классам)
        self.img_id_to_anns = {}
        for ann in self.annotations:
            if ann["category_id"] not in self.catid2idx:
                # например, nest / safety_sign+, если мы их отфильтровали
                continue
            img_id = ann["image_id"]
            self.img_id_to_anns.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.images_root, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        rotations = []

        for ann in anns:
            boxes.append(ann["bbox"])  # [x, y, w, h] в COCO
            labels.append(self.catid2idx[ann["category_id"]])
            areas.append(ann.get("area", 0.0))
            iscrowd.append(ann.get("iscrowd", 0))
            if "rotation" in ann:
                rotations.append(ann["rotation"])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            # картинка без объектов
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if rotations:
            target["rotation"] = torch.tensor(rotations, dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def decode_labels(self, labels_or_target):
        """
        Быстрый способ получить имена классов.
        Принимает либо target (dict), либо labels (Tensor / list).
        """
        # если передали весь target
        if isinstance(labels_or_target, dict):
            labels = labels_or_target["labels"]
        else:
            labels = labels_or_target

        # Tensor -> list[int]
        if torch.is_tensor(labels):
            labels = labels.tolist()

        return [self.idx2name.get(int(l), f"unknown_{l}") for l in labels]

    def debug_print_image_classes(self, idx: int):
        """
        Один вызов — сразу посмотреть, какие классы есть на картинке.
        """
        _, target = self[idx]
        names = self.decode_labels(target)
        print(f"image idx={idx}")
        print("  label indices:", target["labels"].tolist())
        print("  class names  :", names)
        return names

if __name__ =="__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_root = os.path.join(project_root, "dataset", "images")
    ann_file = os.path.join(project_root, "dataset", "annotation_data.json")
    train_data = PowerLineCocoDataset(images_root, ann_file)
    test_data = PowerLineCocoDataset(images_root, ann_file)
    img, target = train_data[1]

    # Вариант 1: получить имена классов из target
    cls_names = train_data.decode_labels(target)
    print(cls_names)

    # Вариант 2: вообще в одну строку по индексу картинки
    train_data.debug_print_image_classes(1)

