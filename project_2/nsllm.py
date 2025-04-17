import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from models.base import BaseModel, perform_forward_pass
from models.lstm import LSTMModule
from models.rnn import RNNModule
from models.transformer import TransformerModule
from sentencepiece import SentencePieceProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn import functional as F

def evaluate(model: BaseModel, tokenizer: SentencePieceProcessor, model_type: str, dataset_path: str, max_output: int = 1):
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")
    criterion = nn.CrossEntropyLoss(ignore_index=3) 
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_bleu = 0.0
    num_samples = 0
    total_loss = 0

    for entry in tqdm(data, desc="Evaluating", unit="sample"):
        prompt: str = entry['prompt']
        expected: str = entry['completion']

        prompt_ids = tokenizer.EncodeAsIds(prompt)
        target_ids = tokenizer.EncodeAsIds(expected)
        
        # Generate output
        generated_text = model.generate(
            prompt=prompt,
            max_output=max_output,
            eos_token_ids=eos_token_ids
        )

        # Tokenize for BLEU
        reference = expected.split()
        candidate = generated_text.split()

        # Calculate BLEU score for the current sample
        bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
        total_bleu += bleu

        # Performing forward pass and getting loss
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        target_tensor = torch.tensor([target_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = perform_forward_pass(model=model, input_ids=input_tensor, target_ids=target_ids, model_type=model_type)
            logits = logits[:, -target_tensor.size(1):, :]  
            logits = logits.view(-1, logits.size(-1))       
            target_tensor = target_tensor.view(-1)           
            loss = criterion.forward(logits, target_tensor)
        total_loss += loss.item()
        num_samples += 1

    avg_bleu = total_bleu / num_samples
    avg_loss = total_loss / num_samples
    avg_loss_tensor = torch.tensor(avg_loss, dtype=torch.float32)
    perplexity = torch.exp(avg_loss_tensor).item()
    print(f"\n--- Average BLEU Score: {avg_bleu:.4f} ---")
    print(f"--- Average Perplexity: {perplexity:.4f} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize text data using SentencePiece.')
    parser.add_argument('--model_type', type=str, required=True, help='The type of model to be trained (rnn, lstm, gru, transformer).')
    parser.add_argument('--data', type=str, help='The number of epochs for early stopping.')
    parser.add_argument('--pt', type=str, help='The number of epochs for early stopping.')
    parser.add_argument('--i', action='store_true', help='Enable verbose output.')
    parser.add_argument('--p', type=str, help='Enable verbose output.')
    args = parser.parse_args()
    model_type = args.model_type
    data = args.data
    model_path = args.pt
    is_inference = args.i
    prompt = args.p

    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'rnn':
            model = RNNModule(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                device=device,
                model_path=model_path
            )
    elif model_type == 'lstm':
        model = LSTMModule(
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device=device,
            model_path=model_path
        )
    elif model_type == 'transformer':
        model = TransformerModule(
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device=device,
            model_path=model_path
        )

    if not is_inference:
        evaluate(model=model, tokenizer=tokenizer, model_type=model_type, dataset_path=data)
    else:
        eos_token_ids = tokenizer.EncodeAsIds("<eos>")  
        generated_text = model.generate(
            prompt=prompt,
            max_output=50,
            eos_token_ids=eos_token_ids 
        )
        print(generated_text)