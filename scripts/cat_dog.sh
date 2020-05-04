#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CUDA_VISIBLE_DEVICES=""

for model in alexnet alexnet_bn ResNet50 ResNet50V2
do
  for lr in 0.001 0.005
    do
    python3.7 ${BASEDIR}/main.py \
      --batch_size=128 \
      --exp_name="${model}-lr${lr}" \
      --dataset="cat_dog" \
      --model="${model}" \
      --mode="train_test" \
      --early_stopping_patience=10 \
      --lr=${lr}
    done
done