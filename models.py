from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINOV2_HUB = {
    ("small", True): "dinov2_vits14_reg",
    ("base", True): "dinov2_vitb14_reg",
    ("large", True): "dinov2_vitl14_reg",
    ("giant", True): "dinov2_vitg14_reg",
    ("small", False): "dinov2_vits14",
    ("base", False): "dinov2_vitb14",
    ("large", False): "dinov2_vitl14",
    ("giant", False): "dinov2_vitg14",
}
_DIMS = {"small": 384, "base": 768, "large": 1024, "giant": 1536}


class DINOv2Backbone(nn.Module):
    def __init__(self, variant: str = "base", use_registers: bool = True, pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        if variant not in _DIMS:
            raise ValueError(f"variant must be one of {list(_DIMS)}")
        self.variant = variant
        self.embed_dim = _DIMS[variant]
        name = _DINOV2_HUB[(variant, use_registers)]
        self.backbone = self._load_backbone(name=name, pretrained=pretrained)

    def _load_backbone(self, name: str, pretrained: bool) -> nn.Module:
        # Preferred path: official DINOv2 torch hub models. This downloads weights once.
        try:
            return torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        except Exception as hub_error:
            # Fallback: timm DINOv2 names are useful in offline environments with cached weights.
            try:
                import timm
                timm_names = {
                    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
                    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
                    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
                    "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2.lvd142m",
                    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
                    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
                    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
                    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
                }
                return timm.create_model(timm_names[name], pretrained=pretrained, num_classes=0)
            except Exception as timm_error:
                raise RuntimeError(
                    "Could not load DINOv2 backbone via torch.hub or timm. "
                    "Check internet/cache and model variant."
                ) from timm_error

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            out: Any = self.backbone.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_clstoken", "x_cls", "cls_token"):
                    if key in out:
                        return out[key]
                if "x_norm_patchtokens" in out:
                    return out["x_norm_patchtokens"].mean(dim=1)
            if torch.is_tensor(out):
                return out[:, 0] if out.ndim == 3 else out
        out = self.backbone(x)
        if isinstance(out, dict):
            return out.get("x_norm_clstoken", next(iter(out.values())))
        return out[:, 0] if out.ndim == 3 else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class MinutiaeFreeDINOv2(nn.Module):
    """Raw fingerprint image -> DINOv2 CLS embedding, plus DINO projection head for SSL."""
    def __init__(self, variant="base", use_registers=True, pretrained=True, out_dim=65536,
                 hidden_dim=2048, bottleneck_dim=256, embed_dim=None):
        super().__init__()
        self.backbone = DINOv2Backbone(variant=variant, use_registers=use_registers, pretrained=pretrained)
        self.embed_dim = embed_dim or self.backbone.embed_dim
        self.head = DINOProjectionHead(self.embed_dim, out_dim, hidden_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def encode(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        emb = self.backbone(x)
        return F.normalize(emb, dim=1) if normalize else emb


def make_teacher(student: MinutiaeFreeDINOv2) -> MinutiaeFreeDINOv2:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def get_model(model_name: str, model_cfg: dict) -> MinutiaeFreeDINOv2:
    if model_name not in ("dinov2_minutiae_free", "minutiae_free"):
        raise ValueError(f"Unknown model name: {model_name}")
    return MinutiaeFreeDINOv2(
        variant=model_cfg.get("variant", "base"),
        use_registers=model_cfg.get("use_registers", True),
        pretrained=model_cfg.get("pretrained", True),
        out_dim=model_cfg.get("out_dim", 65536),
        hidden_dim=model_cfg.get("hidden_dim", 2048),
        bottleneck_dim=model_cfg.get("bottleneck_dim", 256),
        embed_dim=model_cfg.get("embed_dim"),
    )
