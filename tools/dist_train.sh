nnodes=2
master_addr="10.96.203.75"

CONFIG=$1
GPUS=8
WORKDIR=$2

PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
/root/anaconda3/envs/py37_mmdet_codetr/bin/python3.7 \
 -m torch.distributed.launch --nproc_per_node=${GPUS} --nnodes=${nnodes} --node_rank=0  --master_port=$PORT  --master_addr=${master_addr} \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --work-dir $WORKDIR
