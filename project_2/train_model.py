import os
import tqdm
import torch
import pickle
import argparse
import torch.nn as nn
from torch import optim, Tensor
from models.rnn import RNNModule
from models.lstm import LSTMModule
from datasetup.dataset import TextDataset
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader, random_split

def dump_pickle(losses: tuple[list, list], model_type: str, output: str) -> None:
    """
    Dumps the training and validation losses to a pickle file.
    
    :param losses(tuple): A tuple containing the training and validation losses.
    """
    path = os.path.join(output, f'{model_type}_losses.pkl')
    with open(path, 'wb') as f:
        pickle.dump(losses, f)        
    print(f"Losses saved at {path}")

def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """
    Collate function to pad sequences in a batch. This function is used to ensure that all sequences in a batch have the same length by padding them with a specified value.
    
    :param batch(list): A list of tuples, where each tuple contains a pair of input and target sequences.
    """
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(sequences=input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(sequences=target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

def split_test_val(test_dataset: TextDataset, batch_size: float, val_percent: float=0.2) -> tuple[DataLoader, DataLoader]:
    """
    Splits the test dataset into training and validation datasets.
    
    :param test_dataset(TextDataset): The test dataset to be split.
    :param val_percent(float): The percentage of the test dataset to be used for validation.
    :return tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
    """
    total_size = len(test_dataset)
    val_size = int(val_percent * total_size)
    test_size = total_size - val_size
    train_subset, val_subset = random_split(test_dataset, [test_size, val_size])
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def train_model(model_type: str, train_file: str, output: str, batch_size: int, learning_rate: float, epochs: int, early_stopping_patience: int) -> tuple[list[float], list[float]]:
    """
    Trains the model using the specified model type.
    
    :param model_type(str): The type of model to be trained ('rnn' or 'lstm').
    :return tuple[list[float], list[float]]: A tuple containing the training and validation losses.
    """
    # Initializing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loading Tokenizer
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()

    # Loading Dataset
    train_dataset = TextDataset(train_file, tokenizer=tokenizer, max_seq_len=128)
    train_loader, val_loader = split_test_val(test_dataset=train_dataset, batch_size=batch_size)
    
    # Initializing Model Architecture & Moving to device
    if model_type == 'rnn':
        model = RNNModule(tokenizer=tokenizer, vocab_size=vocab_size).to(device) 
    elif model_type == 'lstm':
        model = LSTMModule(tokenizer=tokenizer, vocab_size=vocab_size).to(device) 
    else:
        raise ValueError(f"Model Type {model_type} is not supported.")

    # Setting up the optimizer, scheduler, and loss function
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3) 

    # Setting training parameters
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []
    
    # Training Loop
    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        
        # Training the model and running on the training dataset
        model.train() 
        for input_ids, target_ids in tqdm.tqdm(train_loader, desc=f"{model_type}_epoch {epoch+1}/{epochs}"):
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
            for input_ids, target_ids in tqdm.tqdm(val_loader, desc=f"{model_type}_valid {epoch+1}/{epochs}"):
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
            if no_improve_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"{model_type}_epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    # Saving the model parameters
    path = os.path.join(output, f'{model_type}.pt')
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    return train_losses, val_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize text data using SentencePiece.')
    parser.add_argument('--model_type', type=str, required=True, help='The type of model to be trained (rnn, lstm, gru, transformer).')
    parser.add_argument('--train', type=str, required=True, help='The path to the training file.')
    parser.add_argument('--output', type=str, required=True, help='Directory to store both the model parameters and losses.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=32, help='The number of epochs for training.')
    parser.add_argument('--early', type=int,  default=3, help='The number of epochs for early stopping.')

    args = parser.parse_args()
    model_type = args.model_type
    train_file = args.train
    output = args.output
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    early_stopping_patience = args.early
    
    train_losses, val_losses = train_model(model_type=model_type,
                                           output=output,
                                           train_file=train_file, 
                                           batch_size=batch_size, 
                                           learning_rate=learning_rate, 
                                           epochs=epochs, 
                                           early_stopping_patience=early_stopping_patience
                                           )
    dump_pickle(losses=(train_losses, val_losses), model_type=model_type, output=output)