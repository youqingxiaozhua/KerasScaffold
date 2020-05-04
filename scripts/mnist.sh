#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CUDA_VISIBLE_DEVICES=""
# test
python3.7 ${BASEDIR}/main.py \
  --batch_size=256 \
  --exp_name="test_cat_dog" \
  --dataset="cat_dog" \
  --model="alexnet" \
  --mode="train_test" \
  --early_stopping_patience=1 \
  --epoch=2

#for model in 'ResNet50' 'ResNet50V2'
#do
#  python ${BASEDIR}/main.py \
#    --batch_size=128 \
#    --exp_name="${model}" \
#    --dataset="mnist" \
#    --model=${model} \
#    --mode="train_test" \
#    --early_stopping_patience=10
#done