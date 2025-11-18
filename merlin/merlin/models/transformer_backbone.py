import torch
import torch.nn as nn

class MerlinTransformerBackbone(nn.Module):
    """
    Simple ViT-style transformer encoder.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        max_tokens: int = 256,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_tokens, d_model) * 0.02
        )

    def forward(self, tokens, attn_mask=None):
        """
        tokens: (B, N, d_model)
        attn_mask: optional (B, N) with 1 for valid, 0 for pad
        """
        B, N, D = tokens.shape
        pos = self.pos_embedding[:, :N, :]
        x = tokens + pos

        # convert attn_mask to transformer format if provided
        src_key_padding_mask = None
        if attn_mask is not None:
            # attn_mask: (B, N) -> bool, True for pad
            src_key_padding_mask = (attn_mask == 0)

        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out
