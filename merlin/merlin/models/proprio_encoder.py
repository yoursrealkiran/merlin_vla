import torch
import torch.nn as nn

class ProprioEncoder(nn.Module):
    """
    Encodes proprioceptive state into a single token.
    """
    def __init__(self, proprio_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, proprio):
        """
        proprio: (B, D)
        returns: (B, 1, d_model)
        """
        token = self.mlp(proprio)  # (B, d_model)
        return token.unsqueeze(1)
