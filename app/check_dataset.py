import os
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T

from da import PowerLineCocoDataset


def collate_fn(batch):
    # стандартный collate для детекции:
    # список картинок, список таргетов
    images, targets = list(zip(*batch))
    return list(images), list(targets)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_root = os.path.join(project_root, "dataset", "images")
    ann_file = os.path.join(project_root, "dataset", "annotation_data.json")

    transform = T.ToTensor()

    full_dataset = PowerLineCocoDataset(
        images_root=images_root,
        ann_file=ann_file,
        transform=transform,
        only_eval_classes=True,  # как у тебя в da.py
    )

    n_total = len(full_dataset)
    print("Всего объектов в датасете:", n_total)

    # пропорции
    test_ratio = 0.15
    val_ratio = 0.15

    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    print(f"train: {n_train}, val: {n_val}, test: {n_test}")

    # фиксируем сид, чтобы сплит был детерминированный
    g = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=g,
    )

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):  ", len(val_dataset))
    print("len(test_dataset): ", len(test_dataset))

    # Дальше сразу можем сделать DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Простейшая проверка — взять один батч с train
    images, targets = next(iter(train_loader))
    print("batch size:", len(images))
    print("target[0] keys:", targets[0].keys())
