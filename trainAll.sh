#!/bin/bash
datasets=( $(cut -d ',' -f3 ./Results.csv ) )

DATA_DIR='/mnt/AI_2TB/UCR/UCRArchive_2018/'
ATCN_MODEL='T0'

for ((i=1; i<${#datasets[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=2 python3.6 train.py --conv2d -mc ${ATCN_MODEL} --data-dir ${DATA_DIR} --dataset ${datasets[$i]} -c checkpoints/T0/${datasets[$i]}  --jitter --magwarp --windowwarp --scaling --augmentation_ratio 1000 --preset-files --ucr2018 --clip-grad -b 256
done
