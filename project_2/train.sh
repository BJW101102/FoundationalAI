#!/bin/bash
source ./.wsl/bin/activate

SCRIPT_PATH="./project_2/train_model.py"
OUTPUT_DIR="./project_2/model_results"
TRAIN_FILE="./project_2/gutenburg/data/train.jsonl"  

# Define model types, epochs, and learning rates
MODEL_TYPES=("transformer" "lstm" "rnn")
EPOCHS=(20 30 30)
LEARNING_RATES=(0.0001 0.001 0.001)

# Clear existing GPU memory
echo "Clearing GPU memory..."
PIDS=$(nvidia-smi | grep "python" | awk '{print $5}')
if [ -n "$PIDS" ]; then
    echo "Killing processes: $PIDS"
    kill -9 $PIDS
    sleep 5
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs."

# Loop over each GPU and launch training with appropriate model type and epoch count
for (( i=0; i<$NUM_GPUS; i++ ))
do
    MODEL_TYPE=${MODEL_TYPES[$i]}
    MODEL_EPOCHS=${EPOCHS[$i]}
    LR=${LEARNING_RATES[$i]}

    if [ -z "$MODEL_TYPE" ]; then
        echo "No model type defined for GPU $i. Skipping."
        continue
    fi

    echo "Launching $MODEL_TYPE on GPU $i with $MODEL_EPOCHS epochs and learning rate $LR..."
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_PATH \
        --model_type $MODEL_TYPE \
        --train "$TRAIN_FILE" \
        --output "${OUTPUT_DIR}" \
        --epochs $MODEL_EPOCHS \
        --lr $LR &
done

wait
echo "All training jobs finished."
