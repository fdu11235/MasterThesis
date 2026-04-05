import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions with optional channel projection."""

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # 1x1 projection if channels change
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ThresholdCNN(nn.Module):
    """1D CNN for predicting optimal k (threshold) from diagnostic features.

    Architecture: stacked ResBlocks → multi-scale adaptive pooling → MLP head.
    """

    def __init__(self, in_channels=4, channels=None, kernel_size=5, dropout=0.2,
                 n_classes=10, task="classification", pool_sizes=None):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 256]
        if pool_sizes is None:
            pool_sizes = [1, 4, 16]

        self.task = task

        # Backbone: stacked residual blocks
        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(ResBlock1d(prev_ch, ch, kernel_size))
            prev_ch = ch
        self.conv = nn.Sequential(*blocks)

        # Multi-scale pooling
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(s) for s in pool_sizes])
        pool_dim = prev_ch * sum(pool_sizes)

        # Head
        if task == "regression":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(pool_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(pool_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
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
        # Multi-scale pooling: pool at different resolutions, concatenate
        pooled = [p(x).flatten(1) for p in self.pools]
        x = torch.cat(pooled, dim=1)
        out = self.head(x)
        if self.task == "regression":
            out = out.squeeze(-1)  # (batch,)
        return out
