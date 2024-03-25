#!/bin/bash

PYTHON_SCRIPT="./Experiment/run.py"
PHASE="final"
EPOCHS=30
BATCHSIZE=32
MODEL="dnn"
DEVICE=1

# Set the log path
LOG_PATH="Logs/${PHASE}/${MODEL}"

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Export GPU Number
export CUDA_VISIBLE_DEVICES=$DEVICE

echo python -u "$PYTHON_SCRIPT"  \
    --epochs $EPOCHS        \
    --batch-size $BATCHSIZE \
    --model $MODEL > "${LOG_PATH}/test.log" 2>&1

python -u "$PYTHON_SCRIPT"  \
    --epochs $EPOCHS        \
    --batch-size $BATCHSIZE \
    --model $MODEL > "${LOG_PATH}/test.log" 2>&1
