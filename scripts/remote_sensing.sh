#!/usr/bin/env bash
#source /home/xuefanglei/venvs/DLHomework/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"
export TF_ENABLE_AUTO_MIXED_PRECISION=1

model="DeepLabV3Plus"
loss="categorical_crossentropy"
lr=0.001
batch=16

python3.7 main.py \
  --batch_size=${batch} \
  --exp_name="${model}-lr${lr}-${loss}" \
  --loss="${loss}" \
  --dataset="remote_sensing" \
  --model="${model}" \
  --mode="train_test" \
  --early_stopping_patience=10 \
  --lr=${lr} \
  --resume='ckpt' \
  --weights='imagenet' \
  --debug \
  --epoch=200

