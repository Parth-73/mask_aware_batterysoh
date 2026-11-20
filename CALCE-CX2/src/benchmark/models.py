"""Identical input information, shared latent-aging hypothesis for both PINNs."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ('lstm', 'lstm_pinn', 'transformer', 'cnn_pinn')


class BenchmarkModel(nn.Module):
    def __init__(self, architecture, input_size=5):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        self.architecture = architecture
        self.physics = architecture.endswith('_pinn')
        if architecture.startswith('lstm'):
            self.encoder = nn.LSTM(input_size, 64, num_layers=1, batch_first=True)
        elif architecture == 'transformer':
            self.embedding = nn.Linear(input_size, 64)
            layer = nn.TransformerEncoderLayer(64, 4, 128, .1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            pos = torch.arange(256).float()[:, None]
            div = torch.exp(torch.arange(0, 64, 2).float() * (-math.log(10000.) / 64))
            encoding = torch.zeros(256, 64)
            encoding[:, 0::2] = torch.sin(pos * div)
            encoding[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('position', encoding)
        else:
            self.encoder = nn.Sequential(nn.Conv1d(input_size, 32, 3, padding=1), nn.GELU(),
                                         nn.Conv1d(32, 64, 3, padding=1), nn.GELU())
        self.head = nn.Linear(64, 2 if self.physics else 1)

    def forward(self, x):
        if self.architecture.startswith('lstm'):
            z = self.encoder(x)[0][:, -1]
        elif self.architecture == 'transformer':
            # All tokens are available charging observations at the current cutoff.
            z = self.encoder(self.embedding(x) + self.position[:x.shape[1]])[:, -1]
        else:
            z = self.encoder(x.transpose(1, 2)).mean(-1)
        h = self.head(z)
        if self.physics:
            degradation = F.softplus(h[:, 0])
            correction = .03 * torch.tanh(h[:, 1])
            return 1.0 - degradation + correction, degradation, correction
        return h[:, 0], None, None


def physics_terms(degradation, correction, previous_degradation, valid_pairs):
    """Soft monotonicity on consecutive observed windows, shared by both PINNs.

    L_phys = mean(max(D_previous-D_current,0)^2).
    L_reg = mean(correction^2). Bound |correction| <= 0.03 SOH fraction.
    No battery-specific PDE, latent-state reset, or future measurement is used.
    """
    reg = correction.float().square().mean()
    if valid_pairs.any():
        phys = F.relu(previous_degradation.float()[valid_pairs] - degradation.float()[valid_pairs]).square().mean()
    else:
        phys = degradation.float().sum() * 0
    return phys, reg
