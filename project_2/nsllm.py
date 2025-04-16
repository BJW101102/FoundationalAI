from tqdm import tqdm
import torch
import json
from models.base import BaseModel, perform_forward_pass
from models.lstm import LSTMModule
from models.rnn import RNNModule
from models.transformer import TransformerModule
from sentencepiece import SentencePieceProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn import functional as F


def evaluate(model: BaseModel, tokenizer: SentencePieceProcessor, model_type: str, dataset_path: str, max_output: int = 1):
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_bleu = 0.0
    total_tokens = 0
    num_samples = 0
    total_loss = 0

    for entry in tqdm(data, desc="Evaluating", unit="sample"):
        prompt: str = entry['prompt']
        expected: str = entry['completion']

        prompt_ids = tokenizer.EncodeAsIds(prompt)
        target_ids = tokenizer.EncodeAsIds(expected)
        
        # # Generate output
        # generated_text = model.generate(
        #     prompt=prompt,
        #     max_output=max_output,
        #     eos_token_ids=eos_token_ids
        # )

        # # Tokenize for BLEU
        # reference = expected.split()
        # candidate = generated_text.split()

        # # Calculate BLEU score for the current sample
        # bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
        # total_bleu += bleu

        # Calculating Perplexity
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        target_tensor = torch.tensor([target_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = perform_forward_pass(model=model, input_ids=input_tensor, target_ids=target_ids, model_type=model_type)
    
            # Since target is only 1 token, match last logit position
            logits = logits[:, -1:, :]  
            target_tensor = target_tensor[:, -1:]  

            # Flatten for loss calculation & calculate
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_tensor.reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat, reduction='sum')

            valid_tokens = target_tensor.numel()  # Should be 1 per sample here
            total_tokens += valid_tokens
        
        total_loss += loss.item()
        num_samples += 1

    # Calculate the average BLEU score
    # avg_bleu = total_bleu / num_samples
    avg_nll = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    # print(f"\n--- Average BLEU Score: {avg_bleu:.4f} ---")
    print(f"--- Average Perplexity: {perplexity:.4f} ---")

if __name__ == '__main__':
    # tokenizer = SentencePieceProcessor()
    # tokenizer.Load(model_file='bpe_tokenizer.model')
    # vocab_size = tokenizer.vocab_size()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = LSTMModule(
    #     tokenizer=tokenizer,
    #     vocab_size=vocab_size,
    #     device=device,
    #     model_path="./project_2/model_results/lstm.pt"
    # )

    # evaluate(model=model, tokenizer=tokenizer, model_type='lstm', dataset_path=r'C:\Users\bwalto8\Documents\GitHub\FoundationalAI\project_2\gutenburg\data\test.jsonl')
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModule(tokenizer=tokenizer, vocab_size=vocab_size, device=device, model_path='./project_2/model_results/lstm.pt')
    prompt = "What do you prefer? Cats or Dogs?"
    max_output = 50  
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")  #
    generated_text = model.generate(
        prompt=prompt,
        max_output=max_output,
        eos_token_ids=eos_token_ids # Fix this
    )

    print(generated_text)