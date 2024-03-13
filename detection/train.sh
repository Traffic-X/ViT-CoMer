kill -9 $(lsof -t /dev/nvidia*)
sleep 1s
sh dist_train.sh configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py 8 --seed 2023