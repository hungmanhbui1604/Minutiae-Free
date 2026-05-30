import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


class SquarePad:
    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        return F.pad(image, (hp, vp, max_wh - w - hp, max_wh - h - vp), fill=255)


class RandomElastic:
    def __init__(self, alpha: float = 5.0, sigma: float = 0.5, p: float = 0.25):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        arr = np.array(image)
        h, w = arr.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        return Image.fromarray(warped)


class FingerprintMorphology:
    """Simulate dry/wet ridge conditions without extracting minutiae."""
    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        arr = np.array(image)
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            arr = cv2.erode(arr, kernel, iterations=1)
        else:
            arr = cv2.dilate(arr, kernel, iterations=1)
        return Image.fromarray(arr)


class RandomGaussianNoise:
    def __init__(self, std_max: float = 0.05, p: float = 0.25):
        self.std_max = std_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * random.uniform(0.0, self.std_max)
        return torch.clamp(tensor + noise, 0.0, 1.0)


class MultiCropDINOTransform:
    def __init__(self, global_size=224, local_size=98, global_crops=2, local_crops=8):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        common_pre = [SquarePad()]
        self.global_transform = T.Compose(common_pre + [
            T.RandomResizedCrop(global_size, scale=(0.65, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.02)], p=0.8),
            T.RandomGrayscale(p=0.1),
            T.RandomRotation(180, fill=255),
            T.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.8, 1.2), fill=255),
            RandomElastic(alpha=5.0, sigma=0.5, p=0.25),
            FingerprintMorphology(p=0.25),
            T.RandomErasing(p=0.25, scale=(0.02, 0.50), ratio=(0.3, 3.3), value=1.0),
        ])
        # RandomErasing works on tensors, so tensor conversion is handled separately.
        self.global_tensor = T.Compose([T.ToTensor(), RandomGaussianNoise(0.05, p=0.25), normalize])
        self.local_transform = T.Compose(common_pre + [
            T.RandomResizedCrop(local_size, scale=(0.20, 0.65), ratio=(0.75, 1.33), antialias=True),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.02)], p=0.8),
            T.RandomGrayscale(p=0.1),
            T.RandomRotation(180, fill=255),
            FingerprintMorphology(p=0.20),
        ])
        self.local_tensor = T.Compose([T.ToTensor(), RandomGaussianNoise(0.05, p=0.20), normalize])
        self.global_crops = global_crops
        self.local_crops = local_crops

    def _global(self, image):
        img = self.global_transform.transforms[:-1][0](image) if False else image
        # Manual sequence because RandomErasing requires tensor.
        for t in self.global_transform.transforms[:-1]:
            image = t(image)
        tensor = self.global_tensor(image)
        tensor = self.global_transform.transforms[-1](tensor)
        return tensor

    def __call__(self, image: Image.Image):
        crops = []
        for _ in range(self.global_crops):
            img = image
            for t in self.global_transform.transforms[:-1]:
                img = t(img)
            ten = self.global_tensor(img)
            ten = self.global_transform.transforms[-1](ten)
            crops.append(ten)
        for _ in range(self.local_crops):
            img = self.local_transform(image)
            crops.append(self.local_tensor(img))
        return crops


def multicrop_collate(batch):
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(n_crops)]


def get_transforms(transform_name: str = "dinov2", crops_cfg: dict | None = None):
    if transform_name != "dinov2":
        raise ValueError(f"Unknown transform_name: {transform_name}")
    crops_cfg = crops_cfg or {}
    train_transform = MultiCropDINOTransform(
        global_size=crops_cfg.get("global_size", 224),
        local_size=crops_cfg.get("local_size", 98),
        global_crops=crops_cfg.get("global_crops", 2),
        local_crops=crops_cfg.get("local_crops", 8),
    )
    eval_transform = T.Compose([
        SquarePad(),
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform, eval_transform
