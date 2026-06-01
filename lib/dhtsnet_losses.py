import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterogeneityDecouplingLoss(nn.Module):
    """Optional decoupling regularization between two heterogeneous representations."""

    def __init__(self, loss_weight: float = 0.2, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, structural_feature: torch.Tensor, heterogeneous_feature: torch.Tensor) -> torch.Tensor:
        structural_global = structural_feature.mean(dim=1)

        loss = off_diag - 0.5 * on_diag
        return self.loss_weight * loss
