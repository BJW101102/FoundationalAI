import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
from torch.types import Number
from typing import Tuple


class BaseNN(ABC, nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,  pad_token_id: int):
        super(BaseNN, self).__init__()
        
        # Converts token IDs into dense vector representations, allowing the model to capture semantic relationships between tokens.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id)       
        
        # Hidden-to-Output Layer: mapping hidden state -> output (Fully Connected/Dense Layer)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)          

        self.nn = None

    @abstractmethod
    def forward(self, input_ids: list[int], hidden_state) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def predict_next_token(self, temperature: int, input_ids: list[int], hidden=None) -> Tuple[Number, Tensor]:
        pass

class RNNModule(BaseNN):

    def __init__(self, vocab_size: int, embed_dim: int=256, hidden_dim: int=512, num_layers: int=6, dropout: float=0.2, pad_token_id: int=0):
        super(RNNModule, self).__init__(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            pad_token_id=pad_token_id
        )
        self.nn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    
    
    def forward(self, input_ids: list[int] | Tensor, prev_hidden: Tensor =None) -> Tuple[Tensor, Tensor]:

        """
        Performs a forward pass at time step ,t, through the RNN network and returns the logits (raw predictions)
        and the current hidden state

        :param input_ids(list[int] | Tensor): Tensor/List of input_ids
        :param prev_hidden: The previous layer's hidden state, default is None for first pass.
        :return forward(tuple[Tensor, Tensor]): The logits and the current hidden state
        """

        # Step 1: Embed the tokens (Transform each token in a vector representation, at first is randomized)
        embeddings = self.embedding.forward(input_ids)
        
        # Step 2: Pass through the NN layers (Model is capturing semantics of the embeddings)
        output, current_hidden = self.nn.forward(input=embeddings, hx=prev_hidden)

        # Step 3: Pass the output and get the logits at this time step (Logits: Raw output)
        logits = self.fc.forward(input=output)

        return logits, current_hidden        

    def predict_next_token(self, temperature: int, input_ids: list[int], hidden=None) -> Tuple[Number, Tensor]:
        self.eval()

        with torch.no_grad():
            logits, hidden = self.forward(input_ids=input_ids)
            logits = logits / temperature

        probabilities = F.softmax(logits, dim=-1)[0, -1]
        predicted_token_id = torch.argmax(probabilities, dim=-1)  # Get the index for the last time step

        return predicted_token_id.item(), hidden