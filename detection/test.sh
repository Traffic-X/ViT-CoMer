kill -9 $(lsof -t /dev/nvidia*)
sleep 1s
bash dist_test.sh configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_3x_coco.py mask_rcnn_dinov2_vit_comer_base_3x.pth 8 --eval bbox