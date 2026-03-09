import torch
import torch.nn as nn


class ThresholdCNN(nn.Module):
    """1D CNN for predicting optimal k (threshold) from diagnostic features."""

    def __init__(self, in_channels=3, channels=None, kernel_size=5, dropout=0.2, n_classes=10):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(ch),
            ])
            prev_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_ch, n_classes),
        )

    def forward(self, x):
        # x: (batch, channels, length)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, last_channel)
        return self.head(x)
