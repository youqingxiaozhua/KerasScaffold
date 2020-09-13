#!/usr/bin/env bash
#source /home/xuefanglei/venvs/DLHomework/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"

#model="ResNet50"
model="EfficientNetB7"
model="InceptionV3"
lr=0.001
batch=64

for model in EfficientNetB7 InceptionV3
do
  for lr in 0.001
    do
    python3 main.py \
      --batch_size=${batch} \
      --exp_name="${model}-lr${lr}" \
      --dataset="cat_dog" \
      --model="${model}" \
      --mode="train_test" \
      --early_stopping_patience=10 \
      --lr=${lr} \
      --resume='ckpt' \
      --weights='imagenet' \
      --debug
    done
done

#python3 main.py \
#  --batch_size=${batch} \
#  --exp_name="${model}-lr${lr}-pretrain" \
#  --dataset="cat_dog" \
#  --model="${model}" \
#  --mode="train_test" \
#  --early_stopping_patience=10 \
#  --lr=${lr} \
#  --weights='imagenet' \
#  --freeze_layers=100 \
#  --debug

#python3 main.py \
#  --batch_size=${batch} \
#  --exp_name="${model}-lr${lr}-pretrain-finetune" \
#  --dataset="cat_dog" \
#  --model="${model}" \
#  --mode="train_test" \
#  --early_stopping_patience=20 \
#  --lr=${lr} \
#  --resume="" \
#  --weights='imagenet' \
#  --debug
