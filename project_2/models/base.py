import torch.nn as nn
from abc import ABC, abstractmethod
from sentencepiece import SentencePieceProcessor

class BaseModel(ABC, nn.Module):
    def __init__(self, tokenizer: SentencePieceProcessor, vocab_size: int, embed_dim: int, fc_in_features: int,  pad_token_id: int):
        super(BaseModel, self).__init__()
         
        # Tokenizer for tokenizing the strings 
        self.tokenizer = tokenizer

        # Converts token IDs into dense vector representations, allowing the model to capture semantic relationships between tokens.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id)       
        
        # Hidden-to-Output Layer: mapping input -> output (Fully Connected/Dense Layer)
        self.fc = nn.Linear(in_features=fc_in_features, out_features=vocab_size)          
