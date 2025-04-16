import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from sentencepiece import SentencePieceProcessor

class BaseModel(ABC, nn.Module):
    def __init__(self, device: str, tokenizer: SentencePieceProcessor, vocab_size: int, embed_dim: int, fc_in_features: int, pad_token_id: int):
        super(BaseModel, self).__init__()

        # Hardware Location
        self.device = device

        # Tokenizer for tokenizing the strings 
        self.tokenizer = tokenizer

        # Converts token IDs into dense vector representations, allowing the model to capture semantic relationships between tokens.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id)       

        # Hidden-to-Output Layer: mapping input -> output (Fully Connected/Dense Layer)
        self.fc = nn.Linear(in_features=fc_in_features, out_features=vocab_size)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Defines the forward pass of the model."""
        pass

    @abstractmethod
    def predict_next_token(self, *args, **kwargs):
        """Predicts the next token given an input sequence."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float = 0.8) -> str:
        """Generates text from the given prompt."""
        pass

def perform_forward_pass(model: BaseModel, input_ids: Tensor, target_ids: Tensor, model_type: str) -> Tensor:
    """
    Performs a forward pass through the model for the correct model_type

    :param model(BaseModel): The model
    :param input_ids(Tensor): The input token ids
    :param model_type(str): The type of model (rnn, lstm, or transformer)
    :return logits(Tensor): The raw output of the model (logits)
    """

    if model_type == 'transformer':
        logits = model.forward(input_ids)
    else:
        logits, _ = model.forward(input_ids)
    return logits