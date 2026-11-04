"""Length-preserving dilation-only counterpart of model.ECGModel."""

import torch.nn as nn


class ECGDilatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, dilation=1, padding=3, bias=True),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4, bias=True),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, dilation=4, padding=8, bias=True),
            nn.ReLU(),
            nn.Conv1d(64, 11, kernel_size=1, dilation=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.network(x)
