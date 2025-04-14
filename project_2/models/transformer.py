import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.types import Number
from sentencepiece import SentencePieceProcessor
from base import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModule(BaseModel):
    def __init__(self, 
        tokenizer: SentencePieceProcessor, 
        vocab_size: int, 
        embed_dim: int = 256, 
        num_heads: int = 8, 
        num_layers: int = 6, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1, 
        pad_token_id: int = 0, 
        device: str = 'cpu'):

        super(TransformerModule, self).__init__(
            tokenizer=tokenizer,
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            fc_in_features=embed_dim, 
            pad_token_id=pad_token_id
        )  

        self.device = device
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to(device)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass through the Transformer model.
        :param input_ids (Tensor): Tensor of shape (batch_size, sequence_length)
        :return: Logits (Tensor of shape (batch_size, sequence_length, vocab_size))
        """
        x = self.embedding.forward(input_ids)
        x = self.pos_encoder.forward(x)
        x = self.model.forward(x)
        logits = self.fc.forward(x)
        return logits

    def predict_next_token(self, temperature: float, input_ids: Tensor) -> Tuple[Number, None]:
        """
        Predicts the next token from the input sequence using temperature-based sampling.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

        probabilities = F.softmax(logits, dim=-1)
        predicted_token_id = torch.multinomial(probabilities, num_samples=1)

        return predicted_token_id.item(), None

    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float = 0.8) -> str:
        """
        Generates a continuation for a given prompt using autoregressive decoding.
        """
        self.eval()
        input_token_ids: list[int] = self.tokenizer.Encode(prompt, out_type=int)
        generated_ids = []

        for _ in range(max_output):
            input_tensor = torch.tensor([input_token_ids], dtype=torch.long, device=self.device)
            next_token_id, _ = self.predict_next_token(temperature, input_tensor)
            if next_token_id in eos_token_ids:
                break
            generated_ids.append(next_token_id)
            input_token_ids.append(next_token_id)

        output = self.tokenizer.Decode(generated_ids, out_type=str)
        return output