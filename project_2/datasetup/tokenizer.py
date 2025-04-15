import os 
import argparse
import sentencepiece as spm

def merge_text_files(input_dir: str, output_file: str):
    """
    Merges all text files in the input directory into a single text file.

    :param input_dir(str): Path to the directory containing text files.
    :param output_file(str): Path to the output text file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + '\n')

def tokenize_data(corpus: str, model_prefix: str, vocab_size: int = 10000) -> str:
    """
    Tokenizes the input corpus using SentencePiece and saves the model.

    :param corpus(str): Path to the input text file containing the corpus.
    :param model_prefix(str): Prefix for the output model files.
    :param vocab_size(int): Size of the vocabulary to be generated.
    :return path(str): Path to the generated model file.
    """

    user_defined_symbols = ",".join(["<bos>", "<eos>", "<pad>"])
    spm.SentencePieceTrainer.Train(
        input=corpus, 
        model_prefix=model_prefix, 
        vocab_size=vocab_size, 
        model_type='bpe', 
        bos_id=1, 
        eos_id=2, 
        pad_id=3, 
        user_defined_symbols=user_defined_symbols
    )
    path = f'{model_prefix}.model'
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize text data using SentencePiece.')
    parser.add_argument('--c', type=str, required=True, help='Directory containing text files to merge and tokenize.')
    parser.add_argument('--o', type=str, required=True, help='Output file for merged text.')
    parser.add_argument('--n', type=str, required=True, help='Prefix for the output SentencePiece model files.')
    args = parser.parse_args()
    corpus_path = args.c
    output_file = args.o   
    model_prefix = args.n

    print("Step 1: Merging Files")
    merge_text_files(input_dir=corpus_path, output_file=output_file)

    print("Step 2: Training Tokenizer")
    model_path = tokenize_data(corpus=output_file, model_prefix=model_prefix)
    
    print(f'Model saved at: {model_path}')

