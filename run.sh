#!/usr/bin/env bash

PROG=$1

BATCHSIZE=$2

ARGS="--seed=1 \
      --gpus=1 \
      --max_epochs=40 \
      --num_workers=4 \
      --train_batch_size=$BATCHSIZE \
      --val_batch_size=$BATCHSIZE \
      --learning_rate=0.03 \
      --weight_decay=4e-4"

python3 $PROG $ARGS
