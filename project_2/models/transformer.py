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
        embed_dim: int = 512, 
        num_heads: int = 8, # Splitting QKV vectors into num_head vector for multi-headed attention
        num_layers: int = 4, # Number of stacked decoder components
        dim_feedforward: int = 1024, # Number of hidden layers in the MLP
        dropout: float = 0.1, 
        pad_token_id: int = 0,
        model_path: str = None):

        super(TransformerModule, self).__init__(
            device=device,
            tokenizer=tokenizer,
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            fc_in_features=embed_dim, 
            pad_token_id=pad_token_id
        )  
        
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        if model_path:
            self.load_state_dict(torch.load(model_path, map_location=device))
        
        self.to(device)
    
    def forward(self, input_ids: Tensor, temperature: float = 0.8) -> Tensor:
        """
        Forward pass through the decoder-only transformer.
        :param input_ids: (batch_size, seq_len)
        :return: logits (batch_size, seq_len, vocab_size)
        """
        # Step 1: Embed the tokens (Transform each token into a dense vector; initially randomized then learned during training)
        embeddings = self.embedding.forward(input_ids)

        # Step 2: Apply positional encoding to retain information about the order of tokens in the sequence
        pos_emb = self.pos_decoder.forward(embeddings)

        # Step 3: Create a causal mask to prevent the model from attending to future positions (autoregressive behavior)
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        ).to(input_ids.device)

        # Step 4: Decode the sequence using the causal attention mask (Purpose: Learn dependencies in the token sequence so far)
        output = self.decoder.forward(tgt=pos_emb, memory=pos_emb, tgt_mask=causal_mask)

        # Step 5: Apply a fully connected (linear) layer to map the decoder output to vocabulary logits
        logits = self.fc.forward(output)

        # Step 6: Apply temperature scaling to adjust the sharpness of the output distribution
        logits = logits / temperature
        
        return logits

    def predict_next_token(self, input_ids: Tensor, temperature: float) -> Tuple[Number, None]:
        """
        Predicts the next token from the input sequence using temperature-based sampling.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, temperature=temperature)
            
            top_k = 50
            probabilities = F.softmax(logits, dim=-1)[0, -1]  # Last token prediction
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
            predicted_token_id = top_k_indices[torch.multinomial(top_k_probs, 1)]

        return predicted_token_id.item()

    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float = 0.8) -> str:
        """
        Generates a continuation for a given prompt using autoregressive decoding.
        """
        self.eval()
        
        # Encode the prompt into token ids using the tokenizer
        input_token_ids = self.tokenizer.Encode(input=prompt, out_type=int)  
        input_tensor = torch.tensor(data=input_token_ids, dtype=torch.long, device=self.device).unsqueeze(dim=0)

        # Initialize the list to hold the generated token ids (starting with the prompt)
        generated_ids: list = input_token_ids
        
        for _ in range(max_output):
            # Only pass the previously generated tokens (not the full prompt)
            # Pass the last generated token as input for autoregressive generation
            target_tensor = torch.tensor([generated_ids[-1:]], dtype=torch.long, device=self.device)

            # Predict the next token using the model (only based on the generated tokens)
            next_token_id = self.predict_next_token(input_ids=target_tensor, temperature=temperature)
            
            # Stop if the next token is in the end-of-sequence token list
            if next_token_id in eos_token_ids:
                break
            
            # Append the next token to the list of generated tokens
            generated_ids.append(next_token_id)
        
        # Decode the generated token ids back to a string
        output = self.tokenizer.Decode(generated_ids, out_type=str)
        return output
