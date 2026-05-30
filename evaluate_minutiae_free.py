import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AuthenticationEvaluationDataset, IdentificationEvaluationDataset, UniqueFingerprintDataset
from metrics import compute_authentication_metrics, compute_identification_metrics
from models import get_model
from transforms import get_transforms


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("teacher", ckpt.get("student", ckpt.get("model", ckpt)))
    # Strip DDP prefixes if necessary.
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    n = len(loader.dataset)
    embs = None
    for idxs, imgs in tqdm(loader, desc="[extract embeddings]", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            z = model.encode(imgs, normalize=True).float()
        if embs is None:
            embs = torch.zeros((n, z.shape[1]), device=device, dtype=torch.float32)
        embs[idxs.to(device)] = z
    return embs


@torch.no_grad()
def evaluate_authentication(pair_loader, embeddings, device):
    all_scores, all_labels = [], []
    for idx_a, idx_b, labels in tqdm(pair_loader, desc="[evaluate pairs]", unit="batch"):
        idx_a = idx_a.to(device); idx_b = idx_b.to(device)
        scores = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_authentication_metrics(np.concatenate(all_scores), np.concatenate(all_labels))


@torch.no_grad()
def evaluate_identification(model, loader, dataset, device):
    model.eval()
    all_embs, all_labels, all_indices = [], [], []
    for imgs, labels, idx in tqdm(loader, desc="[identification inference]", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            z = model.encode(imgs, normalize=True).float().cpu()
        all_embs.append(z); all_labels.extend(labels.numpy()); all_indices.extend(idx.numpy())
    all_embs = torch.cat(all_embs).numpy()
    all_labels = np.array(all_labels); all_indices = np.array(all_indices)
    order = np.argsort(all_indices)
    all_embs, all_labels = all_embs[order], all_labels[order]
    n_gal = dataset.n_gallery
    gallery_embs, gallery_labels = all_embs[:n_gal], all_labels[:n_gal]
    probe_embs, probe_labels = all_embs[n_gal:], all_labels[n_gal:]
    sim_mat = np.dot(probe_embs, gallery_embs.T)
    return compute_identification_metrics(sim_mat, probe_labels, gallery_labels)


def main(cfg, checkpoint_path, split_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _, eval_transform, _ = get_transforms(cfg["data"].get("transform_name", "dinov2"), cfg.get("crops", {}))

    auth_dataset = AuthenticationEvaluationDataset(
        split_path=split_path,
        split="test",
        n_genuine_impressions=cfg["data"].get("n_genuine_impressions", 32),
        n_impostor_impressions=cfg["data"].get("n_impostor_impressions", 1),
        impostor_mode=cfg["data"].get("impostor_mode", "all"),
        n_impostor_subset=cfg["data"].get("n_impostor_subset"),
        seed=cfg["general"].get("seed", 42),
    )
    unique_dataset = UniqueFingerprintDataset(auth_dataset.idx_to_path, transform=eval_transform)
    id_dataset = IdentificationEvaluationDataset(
        split_path=split_path,
        split="test",
        gallery_per_id=cfg["data"].get("gallery_per_id", 1),
        probe_per_id=cfg["data"].get("probe_per_id"),
        transform=eval_transform,
        seed=cfg["general"].get("seed", 42),
    )
    print(auth_dataset)
    print(id_dataset)

    batch_size = cfg.get("evaluation", {}).get("batch_size", 256)
    workers = cfg["training"].get("num_workers", 4)
    pin = cfg["training"].get("pin_memory", True)
    auth_loader = DataLoader(auth_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    unique_loader = DataLoader(unique_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    model = get_model(cfg["model"].get("model_name", "dinov2_minutiae_free"), cfg["model"]).to(device)
    load_weights(model, checkpoint_path, device)
    model.eval()

    embeddings = extract_embeddings(model, unique_loader, device)
    auth_metrics = evaluate_authentication(auth_loader, embeddings, device)
    id_metrics = evaluate_identification(model, id_loader, id_dataset, device)

    print("\nAuthentication")
    print(f"EER: {auth_metrics['eer']:.2%} | AUC: {auth_metrics['auc']:.4f}")
    print(f"TAR@FAR=10%: {auth_metrics['tar_at_far_0.1']:.2%}")
    print(f"TAR@FAR=1% : {auth_metrics['tar_at_far_0.01']:.2%}")
    print(f"TAR@FAR=.1%: {auth_metrics['tar_at_far_0.001']:.2%}")
    print("Identification")
    print(f"Rank-1: {id_metrics['rank_1']:.2%} | Rank-5: {id_metrics['rank_5']:.2%} | Rank-10: {id_metrics['rank_10']:.2%}")

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "split_path": split_path,
        "checkpoint_path": checkpoint_path,
        "authentication": {
            "n_pairs": len(auth_dataset),
            "n_genuine": auth_dataset.n_genuine,
            "n_impostor": auth_dataset.n_impostor,
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in auth_metrics.items() if k not in ("thresholds", "fmr", "tar", "fnmr")},
        },
        "identification": id_metrics,
    }
    out_path = os.path.join(output_dir, "minutiae_free_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_minutiae_free.yaml")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-dir", default="results_minutiae_free")
    args = parser.parse_args()
    main(load_config(args.config), args.checkpoint_path, args.split_path, args.output_dir)
