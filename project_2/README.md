# Not So Large Language Models

This project provides a framework for training and evaluating multiple sequence models (RNN, LSTM, Transformer) for text generation. It includes tools for tokenization, model training, loss visualization, evaluation, and inference.

## Getting Started

This project uses command-line arguments for all scripts. Please ensure you're passing the appropriate arguments when running each script.

# Step 1. Create Vocabulary
To create the vocabulary and tokenizer, run the following script: 
```python
python datasetup/tokenizer.py 
```
This script will save the tokenizer and vocabulary, which are required for training the models.

# Step 2. Train the Model
To train a single model, run the python script directly
```python
python train_model.py 
```

If you'd like to train all three models (RNN, LSTM, and Transformer) sequentially and have 3 NVIDIA GPUs available, use the following command:
```bash
bash train.sh 
```

Both scripts will save the weights and loss history in the specified output directory, with the loss histories stored as serialized .pkl files.

# Step 3. Visualize Loss
To visualize the training losses, open the following notebook:
```bash
datasettup/graphloss.ipynb
```
This notebook loads the saved loss files and generates training loss curves for analysis.

# Step 4. Evaluation
After training, evaluate the model using BLEU and Perplexity scores by running:
```
python nsllm.py 
```

# Step 5. Inference
To generate text completions from a trained model, run the following script with the desired prompt:
```
python nsllm.py --i --q <your_prompt>
```
This will return a generated sequence based on your input prompt.

