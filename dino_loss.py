import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.04,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs: list[torch.Tensor], teacher_outputs: list[torch.Tensor]) -> torch.Tensor:
        student_out = [s / self.student_temp for s in student_outputs]
        teacher_out = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_outputs]
        total_loss, n_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                total_loss += loss
                n_terms += 1
        self.update_center(teacher_outputs)
        return total_loss / max(n_terms, 1)

    @torch.no_grad()
    def update_center(self, teacher_outputs: list[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_outputs, dim=0).mean(dim=0, keepdim=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center /= torch.distributed.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)
