#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29301}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
py37_meta_pd-2.3.0_cu11_comer/bin/python3.7 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --deterministic ${@:3}
