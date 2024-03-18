kill -9 $(lsof -t /dev/nvidia*)
sleep 1s

# vit-comer-tiny
sh dist_train.sh configs/ade20k/upernet_vit_comer_tiny_512_160k_ade20k.py 8 --seed 2023

# vit-comer-small
sh dist_train.sh configs/ade20k/upernet_vit_comer_small_512_160k_ade20k.py 8 --seed 2023

# vit-comer-base
sh dist_train.sh configs/ade20k/upernet_vit_comer_base_512_160k_ade20k.py 8 --seed 2023


