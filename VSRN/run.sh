#!/bin/bash

DATA_PATH=/Users/shaoyangguang/Desktop/CrossModal/VSRN-master/data/f30k_precomp

CUDA_VISIBLE_DEVICES=3
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --lr_update 10  --max_len 60 --batch_size 8