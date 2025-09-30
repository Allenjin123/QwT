#!/bin/bash

mkdir -p logs

# Run zero-shot evaluation
CUDA_VISIBLE_DEVICES=1 python main.py \
    --train-data="/path/to/your/cc3m/cc3m-train-{0000..0575}.tar" \
    --dataset-type "webdataset" \
    --model ViT-B/32 \
    --imagenet-val /path/to/your/ImageNet/val \
    --batch-size 128 \
    --iter 4 \
    --wq_params 6 \
    --aq_params 6 \
    --qwerty
