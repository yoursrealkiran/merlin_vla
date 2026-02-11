import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStem(nn.Module):
    """
    Lightweight CNN stem to process images into a feature map.
    """
    def __init__(self, in_channels=3, hidden_dims=(32, 64, 128)):
        super().__init__()
        layers = []
        c = in_channels
        for h in hidden_dims:
            layers.append(nn.Conv2d(c, h, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(h))
            layers.append(nn.ReLU(inplace=True))
            c = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W)
        return self.net(x)  # (B, C', H', W')


class ImageTokenizer(nn.Module):
    """
    CNN stem + patch flattening + linear projection to token embeddings.
    """
    def __init__(self, d_model: int, in_channels: int = 3, patch_size: int = 2):
        super().__init__()
        self.stem = ConvStem(in_channels=in_channels)
        self.patch_size = patch_size
        self.proj = None
        self.d_model = d_model

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N_tokens, d_model)
        """
        feats = self.stem(x)  # (B, C', H', W')
        B, C, H, W = feats.shape

        # Make sure H, W divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Feature map size must be divisible by patch_size."

        ph = self.patch_size
        pw = self.patch_size
        # (B, C, H//ph, ph, W//pw, pw)
        feats = feats.view(B, C, H // ph, ph, W // pw, pw)
        # move patch dims next to each other: (B, H//ph, W//pw, C, ph, pw)
        feats = feats.permute(0, 2, 4, 1, 3, 5).contiguous()
        # flatten spatial inside patch and channels: (B, N_patches, C*ph*pw)
        feats = feats.view(B, -1, C * ph * pw)

        if self.proj is None:
            # lazily initialize projection based on C*ph*pw
            in_dim = feats.size(-1)
            # ensure projection is placed on same device as feats
            self.proj = nn.Linear(in_dim, self.d_model).to(feats.device)

        tokens = self.proj(feats)  # (B, N_patches, d_model)
        return tokens
