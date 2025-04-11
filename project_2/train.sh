#!/bin/bash
source ./.wsl/bin/activate

SCRIPT_PATH="./project_2/train_model.py"
OUTPUT_DIR="./project_2/model_results"
TRAIN_FILE="./project_2/gutenburg/data/train.jsonl"  

# Define model types based on number of GPUs
MODEL_TYPES=("rnn" "lstm" "gru" "transformer")

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

# Loop over each GPU and launch training with appropriate model type
for (( i=0; i<$NUM_GPUS; i++ ))
do
    MODEL_TYPE=${MODEL_TYPES[$i]}
    if [ -z "$MODEL_TYPE" ]; then
        echo "No model type defined for GPU $i. Skipping."
        continue
    fi

    echo "Launching $MODEL_TYPE on GPU $i..."

    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_PATH \
        --model_type $MODEL_TYPE \
        --train "$TRAIN_FILE" \
        --output "${OUTPUT_DIR}" 
    echo "Started training $MODEL_TYPE on GPU $i → Output: ${OUTPUT_DIR}/gpu_$i"
done

echo "All training jobs launched."
