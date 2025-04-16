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
        
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if model_path:
            self.load_state_dict(torch.load(model_path, map_location=device))
        
        self.to(device)
    
    def forward(self, input_ids: Tensor, temperature: float = 0.8) -> Tensor:
        """
        Forward pass through the decoder-only transformer.
        :param input_ids: (batch_size, seq_len)
        :return: logits (batch_size, seq_len, vocab_size)
        """
        # Step 1: Convert input token IDs into dense embeddings (learned during training)
        embeddings = self.embedding.forward(input_ids)

        # Step 2: Add positional encoding to provide order information to the model
        pos_emb = self.pos_encoder.forward(embeddings)

        # Step 3: Create a causal attention mask to prevent the model from attending to future tokens
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        ).to(input_ids.device)

        # Step 4: Pass the embeddings through stacked Transformer blocks with causal masking
        output = self.encoder.forward(pos_emb, mask=causal_mask)

        # Step 5: Project the final hidden states to vocabulary logits using a linear layer
        logits = self.fc.forward(output)

        # Step 6: Scale the logits by temperature to control randomness during sampling
        logits = logits / temperature
        
        return logits

    def predict_next_token(self, input_ids: Tensor, temperature: float) -> Tuple[Number, None]:
        """
        Predicts the next token from the input sequence using temperature-based sampling.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, temperature=temperature)
            #now we grab the last token of the sequence
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            #sort tokens from highest to lowest probability
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            #cumulative sum of the sorted probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            #the smallest set where cumulative prob exceeds top_p
            cutoff = cumulative_probs > 0.9
            if torch.any(cutoff):
                cutoff_index = torch.min(torch.where(cutoff)[1]) + 1
            else:
                cutoff_index = sorted_probs.shape[-1]
           
            # Slice to get the nucleus set
            filtered_probs = sorted_probs[:, :cutoff_index]
            filtered_indices = sorted_indices[:, :cutoff_index]

            # Re-normalize
            filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)

            sampled_token_idx = torch.multinomial(filtered_probs, num_samples=1)
            predicted_token_id = filtered_indices[0, sampled_token_idx]
            
        return predicted_token_id.item()

    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float = 0.8) -> str:
        """
        Generates a continuation for a given prompt using autoregressive decoding.
        Only the newly generated text (excluding the prompt) is returned.
        """
        self.eval()
        
        # Encode the prompt into token ids using the tokenizer
        input_token_ids = self.tokenizer.Encode(input=prompt, out_type=int)

        # Initialize the list to hold the generated token ids (starting with the prompt)
        generated_ids: list = input_token_ids.copy()
        
        for _ in range(max_output):
            target_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

            # Predict the next token using the model
            next_token_id = self.predict_next_token(input_ids=target_tensor, temperature=temperature)
            
            if next_token_id in eos_token_ids:
                break
            
            generated_ids.append(next_token_id)
        
        # Slice off the original prompt to return only the newly generated text
        new_token_ids = generated_ids[len(input_token_ids):]
        output = self.tokenizer.Decode(new_token_ids, out_type=str)        
        return output
