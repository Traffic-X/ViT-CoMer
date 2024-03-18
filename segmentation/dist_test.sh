#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29510}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
py37_meta_pd-2.3.0_cu11_comer/bin/python3.7 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
