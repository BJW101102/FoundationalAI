from tqdm import tqdm
import torch
import json
from models.base import BaseModel
from models.lstm import LSTMModule
from models.rnn import RNNModule
from models.transformer import TransformerModule
from sentencepiece import SentencePieceProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate(model: BaseModel, tokenizer: SentencePieceProcessor, dataset_path: str, max_output: int = 1):
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_bleu = 0.0
    num_samples = 0

    for entry in tqdm(data, desc="Evaluating", unit="sample"):
        prompt: str = entry['prompt']
        expected: str = entry['completion']
        
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
        num_samples += 1

        # # Print the results for each sample
        # print(f"\nPrompt: {prompt}")
        # print(f"Expected: {expected}")
        # print(f"Generated: {generated_text}")
        # print(f"BLEU: {bleu:.4f}")

    # Calculate the average BLEU score
    avg_bleu = total_bleu / num_samples
    print(f"\n--- Average BLEU Score: {avg_bleu:.4f} ---")

if __name__ == '__main__':
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModule(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
        model_path="./project_2/model_results/lstm.pt"
    )

    evaluate(model, tokenizer, dataset_path=r'C:\Users\bwalto8\Documents\GitHub\FoundationalAI\project_2\gutenburg\data\test.jsonl')
