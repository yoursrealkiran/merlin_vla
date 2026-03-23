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
        batch['image']: (B, T, C, H, W) - T is the history window (e.g., 3 frames)
        batch['proprio']: (B, T, D)
        batch['instruction']: list[str] (B,) - Global instruction doesn't change over T
        """
        img_seq = batch["image"]      # (B, T, C, H, W)
        proprio_seq = batch["proprio"] # (B, T, D)
        texts = batch["instruction"]

        B, T, C, H, W = img_seq.shape

        # 1. Process Image Sequence
        # Flatten B and T to process all frames through the CNN stem at once
        img_flat = img_seq.view(B * T, C, H, W)
        img_tokens = self.image_tokenizer(img_flat) # (B*T, N_img, d_model)
        img_tokens = img_tokens.view(B, T * img_tokens.size(1), -1) # (B, T*N_img, d_model)

        # 2. Process Proprio Sequence
        proprio_flat = proprio_seq.view(B * T, -1)
        proprio_tokens = self.proprio_encoder(proprio_flat) # (B*T, 1, d_model)
        proprio_tokens = proprio_tokens.view(B, T, -1) # (B, T, d_model)

        # 3. Text tokens (remain global context)
        text_tokens, text_mask = self.text_encoder(texts)

        # 4. Readout token
        readout = self.readout_token.expand(B, -1, -1)

        # 5. Concatenate everything
        # Order: [Readout, Proprio_History, Text, Image_History]
        tokens = torch.cat([readout, proprio_tokens, text_tokens, img_tokens], dim=1)

        # 6. Mask Generation 
        device = tokens.device
        
        # All extra tokens are valid (1)
        readout_mask = torch.ones(B, 1, device=device, dtype=torch.long)
        proprio_mask = torch.ones(B, T, device=device, dtype=torch.long)
        
        # Calculate total image tokens across the history window
        N_img_total = img_tokens.size(1) 
        img_mask = torch.ones(B, N_img_total, device=device, dtype=torch.long)

        # Concatenate masks in the EXACT same order as the tokens
        # Order: [Readout, Proprio_History, Text, Image_History]
        attn_mask = torch.cat(
            [readout_mask, proprio_mask, text_mask.to(device), img_mask],
            dim=1
        )  # Final Shape: (B, 1 + T + N_txt + (T*N_img))
        
        # 6. Transformer Backbone, generate masks and pass through backbone as before 
        encoded = self.backbone(tokens, attn_mask=attn_mask)
        # 7. Action Head
        return self.action_head(encoded[:, 0, :])