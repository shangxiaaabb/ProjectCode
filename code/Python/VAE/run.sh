#!/bin/bash

MODEL_LIST=("pixelcnn" "pixelcnn_plusplus" "gatedpixelcnn")

for model in "${MODEL_LIST[@]}"
do
  echo "==== Training with model: $model ===="
  CUDA_VISIBLE_DEVICES=1 python VQ_VAE.py --model_type "$model"

  echo "==== Finished training: $model ===="
  echo "==== Cleaning GPU memory ===="
  sleep 10
done
