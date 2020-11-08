#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 train.py -d $2 --conv2d -c $3 $4 $5
tput cnorm
