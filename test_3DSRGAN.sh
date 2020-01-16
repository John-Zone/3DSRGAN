#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result_urban/ \
    --summary_dir ./result_urban/log/ \
    --mode test \
    --is_training False \
    --task SRGAN \
    --batch_size 8 \
    --input_dir_LR ./data/test_LR/ \
    --input_dir_HR ./data/test_HR/ \
    --num_resblock 8 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./1102_SRResnet_01/model-70000 \

