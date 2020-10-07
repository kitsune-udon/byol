PROG=$1
TOTAL_BS=$2
COMMON_ARGS="--distributed_backend=dp \
              --num_nodes=1 \
              --seed=1 \
              --gpus=2 \
              --max_epochs=100 \
              --num_workers=4 \
              --train_batch_size=$TOTAL_BS \
              --val_batch_size=$TOTAL_BS \
              --sync_batchnorm=1"
MASTER_ADDR=localhost MASTER_PORT=12345 \
  python3 $PROG $COMMON_ARGS
