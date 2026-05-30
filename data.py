import glob
import json
import os
import random
from itertools import combinations
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


def _extract_id(path: str, id_type: str = "subject") -> str:
    assert id_type in ("subject", "finger")
    norm = path.replace("\\", "/")
    filename = os.path.basename(norm)

    if "CASIA-FSA" in norm:
        parts = filename.split("_")
        subject_id, finger_id, dst = parts[1][:-1], parts[1][-1], "casiafsa"
    elif "CASIA-FV5" in norm:
        subject_id, finger_id, dst = filename.split("_")[:2] + ["casiafv5"]
    elif "FVC" in norm:
        path_parts = norm.split("/")
        year, db = path_parts[-3], path_parts[-2]
        finger_id = filename.split("_")[0]
        subject_id, dst = finger_id, f"{year}_{db}"
    elif "Neurotechnology-CrossMatch" in norm:
        subject_id, finger_id, dst = filename.split("_")[:2] + ["neurocm"]
    elif "Neurotechnology-UareU" in norm:
        subject_id, finger_id, dst = filename.split("_")[:2] + ["neurouau"]
    elif "PolyU" in norm:
        finger_id = filename.split("_")[0]
        subject_id, dst = finger_id, "polyu"
    elif "SD301a" in norm or "SD301" in norm:
        parts = filename.split("_")
        subject_id, finger_id, dst = parts[0], parts[-1].split(".")[0], "sd301a"
    elif "SD302" in norm:
        parts = filename.split("_")
        subject_id, finger_id, dst = parts[0], parts[-1].split(".")[0], "sd302"
    elif "ATVS-FF" in norm:
        subject_id, finger_id, dst = filename.split("_")[:2] + ["atvsff"]
    else:
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        subject_id = parts[0]
        finger_id = parts[1] if len(parts) > 1 else parts[0]
        dst = os.path.basename(os.path.dirname(path)) or "generic"

    return f"{dst}_{subject_id}" if id_type == "subject" else f"{dst}_{subject_id}_{finger_id}"


def create_recog_splits(
    data_root: str,
    output_path: str,
    split_ratio: tuple = (0.6, 0.2, 0.2),
    min_samples: Optional[int] = 3,
    seed: int = 42,
) -> dict:
    exts = ("*.bmp", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    all_paths = [p for ext in exts for p in glob.glob(os.path.join(data_root, "**", ext), recursive=True)]
    subject_finger_paths: dict[str, dict[str, list[str]]] = {}
    for path in all_paths:
        subject, finger = _extract_id(path, "subject"), _extract_id(path, "finger")
        subject_finger_paths.setdefault(subject, {}).setdefault(finger, []).append(path)

    if min_samples is not None:
        subject_finger_paths = {
            s: {f: ps for f, ps in fingers.items() if len(ps) >= min_samples}
            for s, fingers in subject_finger_paths.items()
        }
        subject_finger_paths = {s: fingers for s, fingers in subject_finger_paths.items() if fingers}

    subjects = sorted(subject_finger_paths)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * split_ratio[0])
    n_val = int(len(subjects) * split_ratio[1])
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])

    splits = {"train": {}, "val": {}, "test": {}, "train_samples": 0, "val_samples": 0, "test_samples": 0}
    for subject, fingers in subject_finger_paths.items():
        split = "train" if subject in train_subjects else "val" if subject in val_subjects else "test"
        for finger, paths in fingers.items():
            splits[split][finger] = sorted(paths)
            splits[f"{split}_samples"] += len(paths)

    for split in ("train", "val", "test"):
        splits[f"{split}_fingers"] = len(splits[split])
    splits["train_subjects"], splits["val_subjects"], splits["test_subjects"] = n_train, n_val, len(subjects) - n_train - n_val
    splits["total_subjects"] = len(subjects)
    splits["total_fingers"] = sum(splits[f"{s}_fingers"] for s in ("train", "val", "test"))
    splits["total_samples"] = sum(splits[f"{s}_samples"] for s in ("train", "val", "test"))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


class FingerprintImageDataset(Dataset):
    """Unlabeled image dataset for DINOv2 self-supervised domain adaptation."""
    def __init__(self, split_path: str, split: str = "train", transform: Optional[Callable] = None):
        with open(split_path, "r") as f:
            self.finger_to_paths = json.load(f)[split]
        self.paths = [p for paths in self.finger_to_paths.values() for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __repr__(self):
        return f"FingerprintImageDataset: {len(self):,} images"


class AuthenticationEvaluationDataset(Dataset):
    def __init__(self, split_path: str, split: str = "test", n_genuine_impressions: int = 32,
                 n_impostor_impressions: int = 1, impostor_mode: str = "all",
                 n_impostor_subset: Optional[int] = None, seed: int = 42):
        assert split in ("val", "test")
        with open(split_path, "r") as f:
            finger_to_paths = json.load(f)[split]
        self.n_ids = len(finger_to_paths)
        path_to_idx = {}
        def get_idx(path: str):
            if path not in path_to_idx:
                path_to_idx[path] = len(path_to_idx)
            return path_to_idx[path]
        rng = random.Random(seed)
        genuine_pairs = []
        for paths in finger_to_paths.values():
            selected = rng.sample(paths, min(len(paths), n_genuine_impressions))
            genuine_pairs += [(get_idx(a), get_idx(b), 1) for a, b in combinations(selected, 2)]
        finger_paths = list(finger_to_paths.values())
        impostor_pairs = []
        if impostor_mode == "all":
            for _ in range(n_impostor_impressions):
                one_per_id = [rng.choice(p) for p in finger_paths]
                impostor_pairs += [(get_idx(a), get_idx(b), 0) for a, b in combinations(one_per_id, 2)]
        else:
            assert n_impostor_subset is not None
            for i, paths in enumerate(finger_paths):
                others = list(range(len(finger_paths)))
                others.remove(i)
                for _ in range(n_impostor_impressions):
                    a = rng.choice(paths)
                    for j in rng.sample(others, n_impostor_subset):
                        impostor_pairs.append((get_idx(a), get_idx(rng.choice(finger_paths[j])), 0))
        self.n_genuine, self.n_impostor = len(genuine_pairs), len(impostor_pairs)
        self.pairs = genuine_pairs + impostor_pairs
        self.idx_to_path = {idx: path for path, idx in path_to_idx.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __repr__(self):
        return f"AuthenticationEvaluationDataset ({self.n_ids:,} ids): {len(self):,} pairs (genuine={self.n_genuine:,}, impostor={self.n_impostor:,})"


class UniqueFingerprintDataset(Dataset):
    def __init__(self, idx_to_path: dict[int, str], transform: Optional[Callable] = None):
        self.idx_to_path = idx_to_path
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_path)

    def __getitem__(self, idx):
        img = Image.open(self.idx_to_path[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return idx, img


class IdentificationEvaluationDataset(Dataset):
    def __init__(self, split_path: str, split: str = "test", gallery_per_id: int = 1,
                 probe_per_id: Optional[int] = None, transform: Optional[Callable] = None, seed: int = 42):
        with open(split_path, "r") as f:
            finger_to_paths = json.load(f)[split]
        self.transform = transform
        self.n_ids = len(finger_to_paths)
        self.gallery_paths, self.gallery_labels, self.probe_paths, self.probe_labels = [], [], [], []
        rng = random.Random(seed)
        id_to_label = {fid: i for i, fid in enumerate(sorted(finger_to_paths))}
        for fid, paths in finger_to_paths.items():
            paths = list(paths)
            rng.shuffle(paths)
            g_paths = paths[:gallery_per_id]
            p_paths = paths[gallery_per_id: gallery_per_id + probe_per_id] if probe_per_id else paths[gallery_per_id:]
            label = id_to_label[fid]
            self.gallery_paths.extend(g_paths); self.gallery_labels.extend([label] * len(g_paths))
            self.probe_paths.extend(p_paths); self.probe_labels.extend([label] * len(p_paths))
        self.all_paths = self.gallery_paths + self.probe_paths
        self.all_labels = self.gallery_labels + self.probe_labels
        self.n_gallery, self.n_probes = len(self.gallery_paths), len(self.probe_paths)

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img = Image.open(self.all_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.all_labels[idx], idx

    def __repr__(self):
        return f"IdentificationEvaluationDataset ({self.n_ids:,} ids): gallery={self.n_gallery:,}, probe={self.n_probes:,}"
