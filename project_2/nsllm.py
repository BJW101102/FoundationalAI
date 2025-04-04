import torch
import json
from torch.utils.data import Dataset
from network import RNNModule, BaseNN
from sentencepiece import SentencePieceProcessor

    


class TextDataset(Dataset):

    def __load_data(self, filepath: str) -> list[list[int]]:
        samples = []
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                prompt = item["prompt"]
                completion = item["completion"]
                text = prompt + " " + completion
                token_ids = self.tokenizer.Encode(input=text, out_type=int)[:self.max_seq_len]
                if len(token_ids) < 2:
                    continue
                samples.append[token_ids]
        return samples


    def __init__(self, filepath: str, tokenizer: SentencePieceProcessor, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self.__load_data(filepath)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        tokens = self.samples[index]
        input_ids = torch.tensor(data=tokens[:-1], dtype=torch.long)        
        target_ids = torch.tensor(data=tokens[1:], dtype=torch.long)        
        return input_ids, target_ids

class NotSoLargeLanguageModel():


    def __init__(self, nn: BaseNN, tokenizer: SentencePieceProcessor):
        self.nn = nn
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_output: int, eos_token_ids: list[int], temperature: float=1.0, device: str='cpu'):
        """
        Generates an output sequence (completion) for a given prompt
        
        :param prompt(str): The input prompt
        :param max_output(int): The maximum tokens generated 
        :param eos_token_ids(list[int]): The list of eos token ids
        :param temperature(int): The temperature setting for sampling (What does this mean??)
        :param device(str): The device to run the model on (cpu | gpu)
        """
        
        self.nn.eval()
        input_token_ids = self.tokenizer.Encode(input=prompt, out_type=int)
        input_tensor = torch.tensor(data=input_token_ids, dtype=torch.long, device=device).unsqueeze(dim=0)
        hidden = None

        generated_ids = []
        for _ in range(max_output):
            next_token_id, hidden = self.nn.predict_next_token(temperature=temperature, input_ids=input_tensor, hidden=hidden)
            if next_token_id in eos_token_ids:
                break

            generated_ids.append(next_token_id)
            input_tensor = torch.tensor(data=[next_token_id], dtype=torch.long, device=device).unsqueeze(dim=0)

        output = self.tokenizer.Decode(input=generated_ids, out_type=str)
        return output



if __name__ == '__main__':
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()

    for i in range(5):
        print(f"ID: {i}, Token: {tokenizer.IdToPiece(i)}")



    rnn = RNNModule(vocab_size=vocab_size)

    model = NotSoLargeLanguageModel(nn=rnn, tokenizer=tokenizer)

    prompt = ""
    max_output = 50  # Max number of tokens to generate
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")  # Specify the EOS token ID, if applicable

    generated_text = model.generate(
        prompt=prompt,
        max_output=max_output,
        eos_token_ids=eos_token_ids # Fix this
    )

    # Print the generated text
    print(generated_text)


