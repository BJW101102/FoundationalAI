import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from sentencepiece import SentencePieceProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

    def compute_perplexity(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the perplexity for a sequence based on model logits and the target.

        :param logits: The model's logits (logits of shape [batch_size, seq_len, vocab_size])
        :param target: The target sequence of token ids (shape [batch_size, seq_len])
        :return: Perplexity score (float)
        """
        # Compute cross-entropy loss (ignoring padding tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='mean')

        # Compute perplexity as the exponential of the cross-entropy loss
        perplexity = torch.exp(loss)

        return perplexity.item()

    def compute_bleu(self, predicted_ids: list[int], reference_ids: list[int], max_n: int = 4) -> float:
        """
        Computes BLEU score between predicted and reference sequences.

        :param predicted_ids: List of predicted token IDs
        :param reference_ids: List of reference token IDs (ground truth)
        :param max_n: The maximum n-gram order to use when computing BLEU
        :return: BLEU score (float between 0 and 1)
        """
        # Convert token IDs to tokens using the tokenizer
        predicted_tokens = [self.tokenizer.IdToPiece([tid])[0] for tid in predicted_ids]
        reference_tokens = [self.tokenizer.IdToPiece([tid])[0] for tid in reference_ids]

        # Compute BLEU score using up to max_n-grams
        weights = tuple(1.0 / max_n for _ in range(max_n))
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [reference_tokens],
            predicted_tokens,
            weights=weights,
            smoothing_function=smoothing
        )

        return bleu_score
