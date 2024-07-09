#!/bin/bash

# This script trains a masked language model using distilbert-base

# Set paths for the training and development datasets
TRAIN_DATA="./dataset/train.txt"
DEV_DATA="./dataset/dev.txt"

# Define the model to be used
MODEL="sentence-transformers/all-mpnet-base-v2"

# Run the training script
CUDA_VISIBLE_DEVICES=2,3 python3 train_mlm.py $MODEL $TRAIN_DATA $DEV_DATA
