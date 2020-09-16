#!/usr/bin/env bash
#source /home/xuefanglei/venvs/DLHomework/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"

model="DeepLabV3PlusUNet"
loss="categorical_crossentropy"
lr=0.006
batch=48
weight_decay=0.00001

python3.7 main.py \
  --batch_size=${batch} \
  --exp_name="${model}-lr${lr}-${loss}-wd${weight_decay}" \
  --loss="dice_loss" \
  --dataset="remote_sensing" \
  --model="${model}" \
  --mode="train" \
  --early_stopping_patience=20 \
  --lr=0.00005 \
  --resume='ckpt' \
  --weights='imagenet' \
  --debug \
  --epoch=150 \
  --weight_decay=0 \
  --early_stopping_monitor="val_mean_io_u" \
  --model_checkpoint_monitor="val_mean_io_u"


# predict
#python3.7 main.py \
#  --batch_size=128 \
#  --exp_name="eval" \
#  --loss="${loss}" \
#  --dataset="remote_sensing" \
#  --model="${model}" \
#  --mode="predict" \
#  --early_stopping_patience=10 \
#  --lr=${lr} \
#  --lr_schedule="poly" \
#  --resume='/root/xiaozhua/dl-homework/dataset/remote_sensing/exp/DeepLabV3Plus-lr0.01-categorical_crossentropy-wd0.00001/ckpt/model-0039.ckpt.h5' \
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
#  --resume='/root/xiaozhua/dl-homework/dataset/remote_sensing/exp/DeepLabV3Plus-lr0.01-categorical_crossentropy-wd0.00001/ckpt/model-0039.ckpt.h5' \
#  --task='visualize_result'

sh scripts/notice.sh

