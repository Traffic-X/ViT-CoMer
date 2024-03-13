kill -9 $(lsof -t /dev/nvidia*)
sleep 1s
sh dist_train.sh configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_3x_coco.py 8 --seed 2023