#!/bin/bash

START_TIME=$(date +%s)
echo "学習開始: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a training.log

# 学習開始
python train.py \
    --dataroot ./dataset/Unpaired_1000 \
    --name marinsnow_removal \
    --model cycle_gan \
    --direction AtoB \
    --batch_size 2 \
    --lr 0.00012 \
    --n_epochs 0 \
    --n_epochs_decay 80 \
    --display_id 0 \
    --display_freq 400 \
    --print_freq 100 \
    --save_epoch_freq 5 \
    --save_latest_freq 5000 \
    --num_threads 4 \
    --verbose

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo "学習終了: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a training.log
echo "所要時間: ${HOURS}時間 ${MINUTES}分 ${SECONDS}秒" | tee -a training.log
