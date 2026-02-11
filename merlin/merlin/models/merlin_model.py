import torch
import torch.nn as nn

from .image_encoder import ImageTokenizer
from .text_encoder import TextEncoder
from .proprio_encoder import ProprioEncoder
from .transformer_backbone import MerlinTransformerBackbone

class MerlinPolicy(nn.Module):
    """
    MERLIN: Multimodal transformer-based policy.
    Inputs: RGB (and optional depth), language, proprio.
    Output: action vector (e.g., Δx, Δy, Δz).
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        text_model_name: str = "t5-small",
        freeze_text_encoder: bool = True,
        proprio_dim: int = 6,
        action_dim: int = 3,
        max_image_tokens: int = 64,
        max_text_tokens: int = 64,
    ):
        super().__init__()

        self.image_tokenizer = ImageTokenizer(d_model=d_model, in_channels=3)
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            d_model=d_model,
            max_length=max_text_tokens,
            freeze_encoder=freeze_text_encoder,
        )
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_dim, d_model=d_model)

        self.backbone = MerlinTransformerBackbone(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_tokens=max_image_tokens + max_text_tokens + 2,  # + proprio + readout
        )

        # learnable readout token
        self.readout_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # simple MLP action head (you can later replace with diffusion head)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, batch):
        """
        batch: dict with keys:
          - 'image': (B, C, H, W) float32
          - 'instruction': list[str] length B
          - 'proprio': (B, D)
        """
        img = batch["image"]       # torch.Tensor
        proprio = batch["proprio"] # torch.Tensor
        texts = batch["instruction"]

        B = img.size(0)

        # 1. image tokens
        img_tokens = self.image_tokenizer(img)  # (B, N_img, d_model)
        N_img = img_tokens.size(1)

        # 2. text tokens
        text_tokens, text_mask = self.text_encoder(texts)  # (B, N_txt, d_model), (B, N_txt)
        N_txt = text_tokens.size(1)

        # 3. proprio token
        proprio_token = self.proprio_encoder(proprio)  # (B, 1, d_model)

        # 4. readout token (same for all, but broadcasted)
        readout = self.readout_token.expand(B, -1, -1)  # (B, 1, d_model)

        # token ordering: [readout, proprio, text, image]
        tokens = torch.cat([readout, proprio_token, text_tokens, img_tokens], dim=1)
        # build attn mask: 1 = valid, 0 = pad
        device = tokens.device
        img_mask = torch.ones(B, N_img, device=device, dtype=torch.long)
        proprio_mask = torch.ones(B, 1, device=device, dtype=torch.long)
        readout_mask = torch.ones(B, 1, device=device, dtype=torch.long)

        attn_mask = torch.cat(
            [readout_mask, proprio_mask, text_mask.to(device), img_mask],
            dim=1
        )  # (B, N_total)

        # 5. transformer backbone
        encoded = self.backbone(tokens, attn_mask=attn_mask)  # (B, N, d_model)

        # 6. extract readout token (index 0)
        readout_out = encoded[:, 0, :]  # (B, d_model)

        # 7. action prediction
        action = self.action_head(readout_out)  # (B, action_dim)
        return action
