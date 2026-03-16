import torch
import torch.nn as nn


class ThresholdCNN(nn.Module):
    """1D CNN for predicting optimal k (threshold) from diagnostic features."""

    def __init__(self, in_channels=4, channels=None, kernel_size=5, dropout=0.2,
                 n_classes=10, task="classification"):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        self.task = task

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

        hidden = prev_ch
        if task == "regression":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(prev_ch, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(prev_ch, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )

    def load_pretrained_backbone(self, state_dict):
        """Copy matching params from pretrained checkpoint.

        Returns (n_loaded, n_skipped).
        """
        own_state = self.state_dict()
        n_loaded = 0
        n_skipped = 0
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                n_loaded += 1
            else:
                n_skipped += 1
        self.load_state_dict(own_state)
        return n_loaded, n_skipped

    def forward(self, x):
        # x: (batch, channels, length)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, last_channel)
        out = self.head(x)
        if self.task == "regression":
            out = out.squeeze(-1)  # (batch,)
        return out
