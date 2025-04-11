import torch
import json
from torch.utils.data import Dataset
from models.rnn import RNNModule
from models.lstm import LSTMModule
from sentencepiece import SentencePieceProcessor



if __name__ == '__main__':
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(model_file='bpe_tokenizer.model')
    vocab_size = tokenizer.vocab_size()


    model = LSTMModule(tokenizer=tokenizer, vocab_size=vocab_size)
    # model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
    # model.to('cpu')

    prompt = "What do you prefer? Cats or Dogs?"
    max_output = 50  # Max number of tokens to generate
    eos_token_ids = tokenizer.EncodeAsIds("<eos>")  # Specify the EOS token ID, if applicable

    generated_text = model.generate(
        prompt=prompt,
        max_output=max_output,
        eos_token_ids=eos_token_ids # Fix this
    )

    # Print the generated text
    print(generated_text)


