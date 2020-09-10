#!/usr/bin/env bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CUDA_VISIBLE_DEVICES="0"
# test
python3 ${BASEDIR}/main.py \
  --batch_size=128 \
  --exp_name="test" \
  --dataset="mnist" \
  --model="lenet" \
  --mode="train_test" \
  --early_stopping_patience=10 \
  --epoch=20 \
  --debug

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