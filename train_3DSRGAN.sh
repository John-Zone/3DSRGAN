#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./0910_SRResnet_01/ \
    --summary_dir ./0910_SRResnet_01/log/ \
    --mode train \
	--pre_trained_model_type SRResnet \
	--pre_trained_model True \
	--check_point 
    --is_training True \
    --task SRGAN \
    --batch_size 8 \
    --flip True \
    --random_crop False \
    --crop_size 24 \
    --num_resblock 8 \
    --perceptual_mode MSE \
    --name_queue_capacity 4096 \
    --image_queue_capacity 4096 \
    --ratio 0.001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
	--learning_rate 0.00005 \
    --max_iter 100000 \
    --queue_thread 10 \
	--save_freq 20000

