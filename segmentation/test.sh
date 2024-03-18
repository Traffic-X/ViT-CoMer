kill -9 $(lsof -t /dev/nvidia*)
sleep 1s

# tiny
sh dist_test.sh configs/ade20k/upernet_vit_comer_tiny_512_160k_ade20k.py /path/to/checkpoint_file 8 --eval mIoU
