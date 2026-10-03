"""Density-conditioned calibration around the frozen EMT social residual model."""
import torch
from torch import nn
from src.models.social_residual import SocialResidualPredictor

NEIGHBOR_LIMIT = 8


def density_scale_from_density(density, calibration):
    scale = 1.+.5*torch.tanh(calibration(density))
    # Float tanh can round to exactly +/-1; keep the implemented scale in the
    # promised open interval while changing no representable interior value.
    eps = torch.finfo(scale.dtype).eps
    return scale.clamp(min=.5+eps, max=1.5-eps)


class DensityAwareSocialResidualPredictor(nn.Module):
    """Scale only the social context from the observed valid-neighbor count."""

    def __init__(self, backbone, seed=42):
        super().__init__()
        self.shared = SocialResidualPredictor(backbone, 'emt_sr', seed)
        if self.shared.gate is not None:
            raise AssertionError('The density model must not reuse the scalar gate')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+30000)
            self.density_calibration = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.density_calibration[-1].weight)
            nn.init.zeros_(self.density_calibration[-1].bias)

    @property
    def backbone(self):
        return self.shared.backbone

    def train(self, mode=True):
        super().train(mode)
        self.shared.train(mode)
        return self

    def forward_context(self, target_context, neighbor_context, relation, mask, base_prediction):
        if mask.ndim != 2 or mask.shape[1] != NEIGHBOR_LIMIT:
            raise ValueError(f'Expected neighbor mask [batch,{NEIGHBOR_LIMIT}]')
        social_context = self.shared.social(target_context, neighbor_context, relation, mask)
        density = mask.sum(dim=1, keepdim=True).to(dtype=target_context.dtype)/NEIGHBOR_LIMIT
        scale = density_scale_from_density(density, self.density_calibration)
        calibrated = scale*social_context
        residual = self.shared.residual(torch.cat((target_context, calibrated), dim=-1)).reshape(-1, 12, 2)
        prediction = base_prediction+residual
        return {
            'future_pred': prediction,
            'base_prediction': base_prediction,
            'residual': residual,
            'social_context': social_context,
            'calibrated_social_context': calibrated,
            'density': density,
            'density_scale': scale,
        }

    def forward(self, target_history, neighbor_history, neighbor_mask, neighbor_relation):
        with torch.no_grad():
            _, target_context = self.backbone.encode(target_history)
            base = self.backbone.decoder(target_context).reshape(-1, 12, 2)
            neighbor_context = torch.zeros((*neighbor_mask.shape, 128), device=target_context.device, dtype=target_context.dtype)
            if neighbor_mask.any():
                _, encoded = self.backbone.encode(neighbor_history[neighbor_mask])
                neighbor_context[neighbor_mask] = encoded
        return self.forward_context(target_context, neighbor_context, neighbor_relation, neighbor_mask, base)
