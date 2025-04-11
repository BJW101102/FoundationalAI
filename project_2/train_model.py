import tqdm
import json
import torch
from torch import optim, Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentencepiece import SentencePieceProcessor
from models.rnn import RNNModule

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 32
EARLY_STOPPING_PATIENCE = 3
VAL_FILE = r'gutenburg/data/test.jsonl'
TRAIN_FILE = r'gutenburg/data/test.jsonl'

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
    
def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(sequences=input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(sequences=target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

def split_test_val(test_dataset: TextDataset, val_percent: float=0.2) -> tuple[DataLoader, DataLoader]:
    total_size = len(test_dataset)
    val_size = int(val_percent * total_size)
    test_size = total_size - val_size
    train_subset, val_subset = random_split(test_dataset, [test_size, val_size])
    train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def train_model(model_type: str):
    # Initializing device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Loading Tokenizer
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()

    # Loading Dataset
    train_dataset = TextDataset(TRAIN_FILE, tokenizer=tokenizer, max_seq_len=128)
    train_loader, val_loader = split_test_val(test_dataset=train_dataset)
    
    # Initializing Model Architecture & Moving to device
    if model_type == 'rnn':
        model = RNNModule(tokenizer=tokenizer, vocab_size=vocab_size).to(device) 
    else:
        print(f"Model Type {model_type} is not found. Exiting")
        return

    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3) 

    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        total_train_loss = 0
        total_val_loss = 0
        
        # Training the model and running on the training dataset
        model.train() 
        for input_ids, target_ids in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # Step 0: Moving input/target to the same device as the model and zeroing gradients
            input_ids: Tensor = input_ids.to(device)
            target_ids: Tensor = target_ids.to(device)
            optimizer.zero_grad()

            # Step 1: Performing a forward pass
            logits: Tensor
            logits, _ = model(input_ids)

            # Step 2: Computing loss gradient
            loss: Tensor = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # Step 3: Performing back propagation
            loss.backward()
            
            # Step 4: Updating Weights and Biases
            optimizer.step()

            # Step 5: Computing training loss
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluating the model and running on the validation dataset
        model.eval()
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids: Tensor = input_ids.to(device)
                target_ids: Tensor = target_ids.to(device)
                logits, _ = model(input_ids)
                val_loss: Tensor = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += val_loss.item()
        
        # Calculating the the loss & adjusting learning rate
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Detecting Early Stoppage
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), f'{model}.pt')
    return train_losses, val_losses

if __name__ == '__main__':
    train_losses, val_losses = train_model(model_type='rnn')

