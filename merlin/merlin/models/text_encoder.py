import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

class TextEncoder(nn.Module):
    """
    Wraps a T5 encoder and projects outputs to d_model.
    """
    def __init__(
        self,
        model_name: str = "t5-small",
        d_model: int = 256,
        max_length: int = 64,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.d_model, d_model)
        self.max_length = max_length

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, texts):
        """
        texts: list[str] of length B
        returns: (B, L_text, d_model)
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.encoder.device)
        attention_mask = enc["attention_mask"].to(self.encoder.device)

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # encoder_hidden_states: (B, L, hidden_dim)
        hidden = out.last_hidden_state
        tokens = self.proj(hidden)  # (B, L, d_model)
        return tokens, attention_mask
