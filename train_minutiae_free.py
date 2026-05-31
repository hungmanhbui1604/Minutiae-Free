import argparse
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import FingerprintImageDataset
from dino_loss import DINOLoss
from models import get_model, make_teacher, update_teacher
from schedulers import get_scheduler
from transforms import get_transforms, multicrop_collate


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        return local_rank, dist.get_world_size()
    return local_rank, 1


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_optimizer(opt_name, parameters, cfg):
    if opt_name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt_name == "adam":
        return torch.optim.Adam(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer: {opt_name}")


def momentum_schedule(base_m, final_m, total_steps):
    for step in range(total_steps):
        yield final_m - (final_m - base_m) * (math.cos(math.pi * step / total_steps) + 1) / 2


def save_checkpoint(path, epoch, student, teacher, optimizer, scheduler, scaler, loss_value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "student": unwrap(student).state_dict(),
        "teacher": unwrap(teacher).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loss": float(loss_value),
    }, path)
    if is_main():
        tqdm.write(f"[checkpoint] saved -> {path}")


def load_checkpoint(path, student, teacher, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    unwrap(student).load_state_dict(ckpt.get("student", ckpt.get("model")), strict=False)
    if "teacher" in ckpt:
        unwrap(teacher).load_state_dict(ckpt["teacher"], strict=False)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1


def train_one_epoch(student, teacher, criterion, loader, optimizer, scheduler, scaler, device, epoch, m_iter, cfg):
    student.train(); teacher.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"[ssl train] epoch {epoch:03d}", unit="batch", leave=False, disable=not is_main())
    for crops in pbar:
        crops = [c.to(device, non_blocking=True) for c in crops]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_out = [teacher(crops[0]), teacher(crops[1])]  # teacher only sees global views
        with torch.autocast(device_type="cuda", enabled=cfg["training"].get("amp", True) and torch.cuda.is_available()):
            student_out = [student(c) for c in crops]
            loss = criterion(student_out, teacher_out)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["training"].get("clip_grad_norm", 3.0))
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["training"].get("clip_grad_norm", 3.0))
            optimizer.step()
        scheduler.step()
        update_teacher(unwrap(student), unwrap(teacher), next(m_iter))
        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(len(loader), 1)


def main(cfg, no_wandb=False, resume=None):
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(cfg["general"]["seed"] + local_rank)
    if is_main():
        print(f"Device: {device} | world_size: {world_size}")

    if is_main() and not no_wandb and cfg["wandb"].get("api_key"):
        wandb.login(key=cfg["wandb"]["api_key"])
        wandb.init(project=cfg["wandb"].get("project", "Minutiae-Free-DINOv2"), config=cfg)

    train_transform, _, _ = get_transforms(cfg["data"].get("transform_name", "dinov2"), cfg.get("crops", {}))
    dataset = FingerprintImageDataset(cfg["data"]["split_path"], split="train", transform=train_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=cfg["general"]["seed"]) if world_size > 1 else None
    local_batch = max(1, cfg["training"]["batch_size"] // world_size)
    loader = DataLoader(dataset, batch_size=local_batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg["training"].get("num_workers", 8), pin_memory=cfg["training"].get("pin_memory", True),
                        drop_last=True, collate_fn=multicrop_collate)
    if is_main():
        print(dataset)

    student = get_model(cfg["model"].get("model_name", "dinov2_minutiae_free"), cfg["model"]).to(device)
    teacher = make_teacher(student).to(device)
    if world_size > 1:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # teacher = DDP(teacher, device_ids=[local_rank], output_device=local_rank)

    criterion = DINOLoss(
        out_dim=cfg["model"].get("out_dim", 65536),
        student_temp=cfg["loss"].get("student_temp", 0.1),
        teacher_temp=cfg["loss"].get("teacher_temp", 0.04),
        center_momentum=cfg["loss"].get("center_momentum", 0.9),
    ).to(device)
    optimizer = get_optimizer(cfg["optimizer"].get("opt_name", "adamw"), student.parameters(), cfg["optimizer"])
    scheduler = get_scheduler(cfg["scheduler"].get("sched_name", "cosine"), optimizer, len(loader), cfg["training"]["epochs"], cfg["scheduler"])
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() and cfg["training"].get("amp", True) else None
    total_steps = len(loader) * cfg["training"]["epochs"]
    m_iter = momentum_schedule(cfg["training"].get("teacher_momentum", 0.994), cfg["training"].get("teacher_momentum_final", 1.0), total_steps)

    start_epoch = 1
    resume = resume or cfg["model"].get("ckpt_path")
    if resume:
        start_epoch = load_checkpoint(resume, student, teacher, optimizer, scheduler, scaler)
        for _ in range((start_epoch - 1) * len(loader)):
            next(m_iter)

    ckpt_dir = cfg["output"].get("checkpoint_dir", "ckpts_minutiae_free")
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        if sampler:
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(student, teacher, criterion, loader, optimizer, scheduler, scaler, device, epoch, m_iter, cfg)
        if is_main():
            print(f"Epoch {epoch:03d} | ssl_loss={avg_loss:.4f}")
            if wandb.run is not None:
                wandb.log({"train/ssl_loss": avg_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
            if epoch % cfg["training"].get("checkpoint_interval", 10) == 0 or epoch == cfg["training"]["epochs"]:
                save_checkpoint(os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:03d}.pth"), epoch, student, teacher, optimizer, scheduler, scaler, avg_loss)
                save_checkpoint(os.path.join(ckpt_dir, cfg["output"].get("best_model_name", "best_minutiae_free.pth")), epoch, student, teacher, optimizer, scheduler, scaler, avg_loss)
        if world_size > 1:
            dist.barrier()
    if is_main() and wandb.run is not None:
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_minutiae_free.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(load_config(args.config), no_wandb=args.no_wandb, resume=args.resume)
