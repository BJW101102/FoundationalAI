import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.types import Number
from sentencepiece import SentencePieceProcessor
from .base import BaseModel

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
        device: str,
        tokenizer: SentencePieceProcessor, 
        vocab_size: int, 
        embed_dim: int = 256, 
        num_heads: int = 8, # Splitting QKV vectors into num_head vector for multi-headed attention
        num_layers: int = 6, # Number of stacked encoder/decoder components
        dim_feedforward: int = 512, # Number of hidden layers in the MLP
        dropout: float = 0.1, 
        pad_token_id: int = 0,
        model_path:str|None=None):

        super(TransformerModule, self).__init__(
            device=device,
            tokenizer=tokenizer,
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            fc_in_features=embed_dim, 
            pad_token_id=pad_token_id
        )  

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        if model_path:
            self.load_state_dict(torch.load(model_path, map_location=device))
        
        self.to(device)
    
    def forward(self, src_ids: Tensor, tgt_ids: Tensor, temperature: float = 0.8) -> Tensor:
        """
        Forward pass through encoder-decoder transformer.
        :param src_ids: (batch_size, src_len)
        :param tgt_ids: (batch_size, tgt_len)
        :return: logits (batch_size, tgt_len, vocab_size)
        """

        # Step 1: Embed the tokens (Transform each token in a vector representation, at first is randomized)
        src_embs = self.embedding.forward(src_ids)
        tgt_embs = self.embedding.forward(tgt_ids)

        # Step 2: Apply positional encoding to retain information about the order of tokens in the sequence
        src_emb = self.pos_encoder.forward(src_embs)
        tgt_emb = self.pos_decoder.forward(tgt_embs)

        # Step 3: Encode the source sequence into memory representation (Purpose: Map all input sequence into an abstract continuous representation that holds the learned information for that whole input sequence)
        memory = self.encoder.forward(src_emb)

        # Step 4: Decode the target sequence using the memory from the encoder (Purpose: Generate text sequence with the learned attention and memory from the encoding layer)
        output = self.decoder.forward(tgt=tgt_emb, memory=memory)

        # Step 5: Apply a fully connected (linear) layer to map outputs to vocabulary logits
        logits = self.fc.forward(output)

        # Step 6: Applying temperature scaling
        logits = logits/temperature

        return logits

    def predict_next_token(self, src_ids: Tensor, tgt_ids: Tensor, temperature:float) -> Tuple[Number, None]:
        """
        Predicts the next token from the input sequence using temperature-based sampling.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(src_ids=src_ids, tgt_ids=tgt_ids, temperature=temperature)
            
            # Apply softmax to get probabilities & sample from the distribution
            probabilities = F.softmax(logits, dim=-1)[0, -1]  # Last timestamp
            predicted_token_id = torch.multinomial(probabilities, num_samples=1)

        return predicted_token_id.item()

    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float = 0.8) -> str:
        """
        Generates a continuation for a given prompt using autoregressive decoding.
        """
        self.eval()
        src_token_ids: list[int] = self.tokenizer.Encode(prompt, out_type=int)
        src_tensor = torch.tensor([src_token_ids], dtype=torch.long, device=self.device)
        generated_ids = []
        tgt_input_ids = [self.tokenizer.bos_id()]  

        for _ in range(max_output):
            target_tensor = torch.tensor([tgt_input_ids], dtype=torch.long, device=self.device)
            next_token_id  = self.predict_next_token(src_ids=src_tensor, tgt_ids=target_tensor, temperature=temperature)
            if next_token_id in eos_token_ids:
                break
            generated_ids.append(next_token_id)
            tgt_input_ids.append(next_token_id)

        output = self.tokenizer.Decode(generated_ids, out_type=str)
        return output