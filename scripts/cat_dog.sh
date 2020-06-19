#!/usr/bin/env bash
#source /home/xuefanglei/venvs/DLHomework/bin/activate
export CUDA_VISIBLE_DEVICES="0,1"

#for model in alexnet alexnet_bn ResNet50 ResNet50V2
#do
#  for lr in 0.001 0.005 0.05
#    do
#    python3 main.py \
#      --batch_size=256 \
#      --exp_name="${model}-lr${lr}" \
#      --dataset="cat_dog" \
#      --model="${model}" \
#      --mode="train_test" \
#      --early_stopping_patience=10 \
#      --lr=${lr} \
#      --debug
#    done
#done

model="ResNet50"
lr=0.05

python3 main.py \
  --batch_size=256 \
  --exp_name="${model}-lr${lr}-augment" \
  --dataset="cat_dog" \
  --model="${model}" \
  --mode="train_test" \
  --early_stopping_patience=30 \
  --lr=${lr} \
  --debug