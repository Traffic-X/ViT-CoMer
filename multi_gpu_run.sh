# #!/usr/bin/env bash

# #--- Multi-nodes training hyperparams ---
# nnodes=2
# master_addr="10.96.203.57"

# # Note:
# # 0. You need to set the master ip address according to your own machines.
# # 1. You'd better to scale the learning rate when you use more gpus.
# # 2. Command: sh scripts/run_train_multinodes.sh node_rank
# ############################################# 
# if [ -f $1 ]; then
#   config=$1
# else
#   echo "need a config file"
#   exit
# fi

# /root/paddlejob/workspace/env_run/lvfeng/anaconda3/envs/py37_mmdet_codetr/bin/python3.7 \
# -m torch.distributed.launch --master_port 29500 --nproc_per_node=8 \
#             --nnodes=${nnodes} --node_rank=0  \
#             --master_addr=${master_addr} \
#             train.py  --config ${config}  ${@:3}


nnodes=2
master_addr="10.96.203.57"

# Note:
# 0. You need to set the master ip address according to your own machines.
# 1. You'd better to scale the learning rate when you use more gpus.
# 2. Command: sh scripts/run_train_multinodes.sh node_rank
############################################# 
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi


/root/paddlejob/workspace/env_run/lvfeng/anaconda3/envs/py37_mmdet_codetr/bin/python3.7 \
-m torch.distributed.launch --master_port 29500 --nproc_per_node=8 \
            --nnodes=${nnodes} --node_rank=0  \
            --master_addr=${master_addr} \
            tools/train.py  --config ${config}  ${@:3}
