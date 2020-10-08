PROG=$1
TOTAL_BS=$2
COMMON_ARGS="--seed=1 \
             --gpus=1 \
             --max_epochs=100 \
             --num_workers=4 \
             --train_batch_size=$TOTAL_BS \
             --val_batch_size=$TOTAL_BS"
MASTER_ADDR=localhost MASTER_PORT=12345 \
  python3 $PROG $COMMON_ARGS
