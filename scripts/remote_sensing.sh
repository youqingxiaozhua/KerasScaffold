#!/usr/bin/env bash
#source /home/xuefanglei/venvs/DLHomework/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"

model="DeepLabV3Plus"
loss="categorical_crossentropy"
lr=0.007
batch=16

python3.7 main.py \
  --batch_size=${batch} \
  --exp_name="${model}-lr${lr}-${loss}-freeze-L2" \
  --loss="${loss}" \
  --dataset="remote_sensing" \
  --model="${model}" \
  --mode="train" \
  --early_stopping_patience=10 \
  --lr=${lr} \
  --resume='ckpt' \
  --weights='imagenet' \
  --debug \
  --epoch=200 \
  --lr_schedule="poly" \
  --freeze_layers=119


# predict
#python3.7 main.py \
#  --batch_size=32 \
#  --exp_name="eval" \
#  --loss="${loss}" \
#  --dataset="remote_sensing" \
#  --model="${model}" \
#  --mode="predict" \
#  --early_stopping_patience=10 \
#  --lr=${lr} \
#  --resume='/root/xiaozhua/dl-homework/dataset/remote_sensing/exp/DeepLabV3Plus-lr0.007-categorical_crossentropy/ckpt/model-0020.ckpt.h5' \
#  --weights='imagenet' \
#  --debug \
#  --epoch=200 \
#  --predict_output_dir='/root/xiaozhua/dl-homework/dataset/remote_sensing/RemoteSensing/results'


# visualize
#python3.7 main.py \
#  --batch_size=1 \
#  --exp_name="eval" \
#  --dataset="remote_sensing" \
#  --model="${model}" \
#  --mode="valid" \
#  --resume='/root/xiaozhua/dl-homework/dataset/remote_sensing/exp/DeepLabV3Plus-lr0.007-categorical_crossentropy/ckpt/model-0020.ckpt.h5' \
#  --task='visualize_result'

sh scripts/notice.sh

