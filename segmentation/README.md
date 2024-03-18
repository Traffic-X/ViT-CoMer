# Applying ViT-CoMer to Semantic Segmentation

Our segmentation code is developed on top of [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).



## Usage

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```

## Main Results and Models

**ADE20K val**





| Method  | Backbone   | Pretrain  | Lr schd | Crop Size | mIoU(SS/MS) | #Param  | Config | Ckpt |Log |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------:|:-------------:|
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/1-2a--MV1yVemzM1QX_0bNQ?pwd=r9uv)                                                                                                 | 1×   | 52.1   | 45.8   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1BMb14R4XaTG0wxbWQWoIGQ?pwd=tkc5)  | [log](https://pan.baidu.com/s/1yW7DoMDTdjeSkNQOA2vwzw?pwd=n62v) |-|

## Evaluation

To evaluate ViT-CoMer-T + UperNet (512) on ADE20k val on a single node with 8 gpus run:

```shell
sh test.sh
```


## Training

To train ViT-CoMer-T + UperNet on ADE20k on a single node with 8 gpus run:

```shell
sh train.sh
```
