import torch
from models.lstm import LSTMModule
from models.rnn import RNNModule
from models.transformer import TransformerModule
from sentencepiece import SentencePieceProcessor

if __name__ == '__main__':
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModule(tokenizer=tokenizer, vocab_size=vocab_size, device=device, model_path="./project_2/model_results/lstm.pt")
    prompt = "Who are you, and what is your purpose?"
    max_output = 50  
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")  #
    generated_text = model.generate(
        prompt=prompt,
        max_output=max_output,
        eos_token_ids=eos_token_ids # Fix this
    )
    print(generated_text)


