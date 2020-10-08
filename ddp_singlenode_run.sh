PROG=$1
TOTAL_BS=$2
COMMON_ARGS="--distributed_backend=ddp \
              --num_nodes=1 \
              --seed=1 \
              --gpus=2 \
              --max_epochs=100 \
              --num_workers=4 \
              --train_batch_size=$(($TOTAL_BS/2)) \
              --val_batch_size=$(($TOTAL_BS/2)) \
              --sync_batchnorm=1 \
              --warmup_epochs=5"
MASTER_ADDR=localhost MASTER_PORT=12345 \
  python3 $PROG $COMMON_ARGS
