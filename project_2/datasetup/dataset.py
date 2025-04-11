import torch
import json
from torch.utils.data import Dataset
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
                samples.append(token_ids)
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