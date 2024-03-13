# Applying ViT-CoMer to Object Detection

Our detection code is developed on top of [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

## Usage

Install [MMDetection v2.22.0](https://github.com/open-mmlab/mmdetection/tree/v2.22.0).

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0
pip install instaboostfast # for htc++
cd ops & sh make.sh # compile deformable attention
```

## Main Results and Models

**Mask R-CNN + DINOv2**

| Method     | Backbone      | Pretrain                                                                                                                                                                        | Lr schd | box AP | mask AP | Config                                                                           | Ckpt | Log                                                                                                                |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------:|
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/10XW-0PYaFw9u3RIvx9UA_g) [提取码: 7ceq]                                                                                                 | 1×   | 52.1   | 45.8   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1_qjX1ce4IgBEADMBCrp-ng) [提取码: o1ej]  | [log](https://pan.baidu.com/s/1yOfzfPwdYoxCWUvjPV8iLg) [提取码: 5qtw] |
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/10XW-0PYaFw9u3RIvx9UA_g) [提取码: 7ceq]                                                                                                 | 3×   | 48.6   | 42.9   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1z4azcHYDrHhAS0X-3X-L2w) [提取码: ngpl]  | [log](https://pan.baidu.com/s/1sRVrnmOi7X2yjkKZSDKg5g) [提取码: weej] |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1GcE_ydcnUbcBxAYWfEymkw) [提取码: odn1]                                                                                                 | 1×   | 52.0   | 45.5   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1cx3QifHI_fJahQZO1NuSkA) [提取码: u4nr]  | [log](https://pan.baidu.com/s/1T0ZRxR7Zkuzvav2yyYFlYQ) [提取码: rbek] |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1GcE_ydcnUbcBxAYWfEymkw) [提取码: odn1]                                                                                                 | 3×   | 54.2   | 47.6   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1ALj_FfMEhIRFHnoulanCAQ) [提取码: spd2]  | [log](https://pan.baidu.com/s/1qGDzlTHiUa3yfhnzuKojiQ) [提取码: sknc] |


## Evaluation

To evaluate Mask-RCNN + ViT-CoMer-B on COCO val2017 on a single node with 8 gpus run:

```shell
bash test.sh
```

## Training

To train Mask-RCNN + ViT-CoMer-B on COCO train2017 on a single node with 8 gpus for 36 epochs run:

```shell
bash train.sh
```