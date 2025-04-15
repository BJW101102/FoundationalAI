import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.types import Number
from sentencepiece import SentencePieceProcessor
from .base import BaseModel

class RNNModule(BaseModel):
    def __init__(self, device: str, tokenizer: SentencePieceProcessor, vocab_size: int, embed_dim: int=256, hidden_dim: int=512, num_layers: int=6, dropout: float=0.2, pad_token_id: int=0, model_path:str|None=None):
        super(RNNModule, self).__init__(
            device=device,
            tokenizer=tokenizer,
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            fc_in_features=hidden_dim, 
            pad_token_id=pad_token_id
        )        

        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout).to(device)
        if model_path:
            self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)



    def forward(self, input_ids: list[int] | Tensor, prev_hidden: Tensor=None, temperature:float=0.8) -> Tuple[Tensor, Tensor]:
        """
        Performs a forward pass through the RNN network and returns the logits (raw predictions)
        and the current hidden state

        :param input_ids(list[int] | Tensor): Tensor/List of input_ids
        :param prev_hidden: The previous layer's hidden state, default is None for first pass.
        :return forward(tuple[Tensor, Tensor]): The logits and the current hidden state
        """

        # Step 1: Embed the tokens (Transform each token in a vector representation, at first is randomized)
        embeddings = self.embedding.forward(input_ids)
        
        # Step 2: Pass through the network layers (Model is capturing semantics of the embeddings)
        output, current_hidden = self.rnn.forward(input=embeddings, hx=prev_hidden)

        # Step 3: Apply a fully connected (linear) layer to map outputs to vocabulary logits
        logits = self.fc.forward(input=output)

        # Step 4: Applying temperature scaling
        logits = logits / temperature


        return logits, current_hidden        

    def predict_next_token(self, temperature: float, input_ids: list[int], hidden=None) -> Tuple[Number, Tensor]:
        self.eval()

        with torch.no_grad():
            logits, hidden = self.forward(input_ids=input_ids, prev_hidden=hidden, temperature=temperature)

            # Apply softmax to get probabilities & sample from the distribution
            probabilities = F.softmax(logits, dim=-1)[0, -1]  # Last timestamp
            predicted_token_id = torch.multinomial(probabilities, num_samples=1)

        return predicted_token_id.item(), hidden
    
    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float=0.8):
        """
        Generates an output sequence (completion) for a given prompt
        
        :param prompt(str): The input prompt
        :param max_output(int): The maximum tokens generated 
        :param eos_token_ids(list[int]): The list of eos token ids
        :param temperature(int): The temperature setting for sampling (What does this mean??)
        :param device(str): The device to run the model on (cpu | gpu)
        """
        
        self.rnn.eval()
        input_token_ids = self.tokenizer.Encode(input=prompt, out_type=int)
        input_tensor = torch.tensor(data=input_token_ids, dtype=torch.long, device=self.device).unsqueeze(dim=0)
        hidden = torch.zeros(self.rnn.num_layers, input_tensor.size(0), self.rnn.hidden_size).to(input_tensor.device)

        generated_ids = []
        for _ in range(max_output):
            next_token_id, hidden = self.predict_next_token(temperature=temperature, input_ids=input_tensor, hidden=hidden)
            if next_token_id in eos_token_ids:
                break

            generated_ids.append(next_token_id)
            input_tensor = torch.tensor(data=[next_token_id], dtype=torch.long, device=self.device).unsqueeze(dim=0)

        output = self.tokenizer.Decode(input=generated_ids, out_type=str)
        return output