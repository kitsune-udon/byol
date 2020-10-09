#!/usr/bin/env bash

PROG=$1

BATCHSIZE=256

ARGS="--seed=1 \
      --gpus=1 \
      --max_epochs=40 \
      --num_workers=4 \
      --train_batch_size=$BATCHSIZE \
      --val_batch_size=$BATCHSIZE \
      --learning_rate=0.2 \
      --weight_decay=1e-6"

python3 $PROG $ARGS
