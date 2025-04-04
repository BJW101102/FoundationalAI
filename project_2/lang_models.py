import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentencepiece import SentencePieceProcessor
from abc import ABC, abstractmethod

class BaseNN(ABC, nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,  pad_token_id: int):
        super(BaseNN, self).__init__()
        
        # Converts token IDs into dense vector representations, allowing the model to capture semantic relationships between tokens.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_token_id)       
        
        # Hidden-to-Output Layer: mapping hidden state -> output (Fully Connected/Dense Layer)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)          

        self.nn = None

    @abstractmethod
    def forward(self, input_ids: list[int], hidden_state):
        pass

    @abstractmethod
    def predict_next_token(self, input_ids: list[int], hidden=None):
        pass


    
class RNNModule(BaseNN):

    def __init__(self, vocab_size: int, embed_dim: int=256, hidden_dim: int=512, num_layers: int=6, dropout: float=0.2, pad_token_id: int=0):
        super(RNNModule, self).__init__(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            pad_token_id=pad_token_id
        )
        self.nn = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    
    
    def forward(self, input_ids: list[int] | Tensor, prev_hidden: Tensor =None) -> tuple[Tensor, Tensor]:

        """
        Performs a forward pass through the RNN network and returns the logits (raw predictions)
        and the current hidden state

        :param input_ids(list[int] | Tensor): Tensor/List of input_ids
        :param prev_hidden: The previous layer's hidden state, default is None for first pass.
        :return forward(tuple[Tensor, Tensor]): The logits and the current hidden state
        """

        # Step 1: Embed the tokens
        embeddings = self.embedding.forward(input_ids)
        
        # Step 2: Pass through the NN layers
        output, current_hidden = self.nn.forward(input=embeddings, hx=prev_hidden)

        # Step 3: Pass the output (Logits: )
        logits = self.fc.forward(input=output)

        return logits, current_hidden        

    def predict_next_token(self, input_ids: list[int], hidden=None):
        self.eval()

        with torch.no_grad():
            logits, hidden = self.forward(input_ids=input_ids)

        probabilities = F.softmax(logits, dim=-1)  # The softmax should be applied to the last dimension (vocab dimension)

        print(probabilities.shape)
        predicted_token_id = torch.argmax(probabilities, dim=-1)

        print(predicted_token_id)


        return predicted_token_id, hidden


class NotSoLargeLanguageModel():

    def __init__(self, nn: BaseNN, tokenizer: SentencePieceProcessor):
        self.nn = nn
        self.tokenizer = tokenizer


    def generate(self, prompt: str, max_output: int, eos_token_id: None, temperature: float, device: str):

        self.nn.eval()

        input_token_ids = self.tokenizer.Encode(input=prompt, out_type=int)
        input_tensor = torch.tensor(data=input_token_ids, dtype=torch.long, device=device).unsqueeze(dim=0)
        hidden = None

        generated_ids = []

        for _ in range(max_output):
            next_token_id, hidden = self.nn.predict_next_token(input_ids=input_tensor, hidden=hidden)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)
            input_tensor = torch.tensor(data=[[next_token_id]], dtype=torch.long, device=device).unsqueeze(dim=0)

        output = self.tokenizer.Decode(input=generated_ids, out_type=str)
        return output



if __name__ == '__main__':


    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    rnn = RNNModule(vocab_size=10000)

    model = NotSoLargeLanguageModel(nn=rnn, tokenizer=tokenizer)

    prompt = "Once upon a time"
    max_output = 50  # Max number of tokens to generate
    eos_token_id = tokenizer.EncodeAsIds("<eos>")  # Specify the EOS token ID, if applicable
    temperature = 1.0  # You can adjust this for creativity (higher = more random)
    device = "cpu"  # Or "cuda" if you are using a GPU

    generated_text = model.generate(
        prompt=prompt,
        max_output=max_output,
        eos_token_id=eos_token_id,
        temperature=temperature,
        device=device
    )

    # Print the generated text
    print(generated_text)


