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
| Mask R-CNN | ViT-S | [DeiT-S](https://pan.baidu.com/s/1BVD24Eeg6S0F2v21mHzI5w?pwd=c4g4)                                                                                                 | 3×   | 44.0  | 39.9 | [config](./configs/mask_rcnn/mask_rcnn_deit_small_fpn_3x_coco.py)         | - | - |
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/1-2a--MV1yVemzM1QX_0bNQ?pwd=r9uv)                                                                                                 | 1×   | 48.6   | 42.9   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1BMb14R4XaTG0wxbWQWoIGQ?pwd=tkc5)  | [log](https://pan.baidu.com/s/1yW7DoMDTdjeSkNQOA2vwzw?pwd=n62v) |
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/1-2a--MV1yVemzM1QX_0bNQ?pwd=r9uv)                                                                                                 | 3×   | 52.1   | 45.8   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1nxgjko_7m_I6OQxEGK__IA?pwd=x5a4)  | [log](https://pan.baidu.com/s/1il2nrRRkRIWv_fVycimn0A?pwd=np4p) |
| Mask R-CNN | ViT-B | [DeiT-B](https://pan.baidu.com/s/1JNknlKiB4lMJsdF-m5ndAQ?pwd=v62p)                                                                                                 | 3×   | 45.8  | 41.3 | [config](./configs/mask_rcnn/mask_rcnn_deit_base_fpn_3x_coco.py)         | - | - |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1gjuuFmYl_cNCc8y7ZE5_rg?pwd=5ngw)                                                                                                 | 1×   | 52.0   | 45.5   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1Nqn3QS2jy0wyn-aKBcbGyg?pwd=derg)  | [log](https://pan.baidu.com/s/1-L9XexL1C8vlrJh9J8X_Yg?pwd=qt9a) |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1gjuuFmYl_cNCc8y7ZE5_rg?pwd=5ngw)                                                                                                 | 3×   | 54.2   | 47.6   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1dUAJ_ToRkNhPrcpqQmgcGQ?pwd=8iam)  | [log](https://pan.baidu.com/s/16byNOInQ1JJ4arjAYtuIMA?pwd=d5ud) |
 
 
 

**ViT-CoMer + Co-DETR**

We combined our ViT-CoMer with the state-of-the-art detection algorithm Co-DETR and achieved excellent results **`64.3 AP`**. In order to help everyone conduct research on this base, we will gradually open up our training configurations and model weights. The specific implementation details, please refer to here [ViT-CoMer+Co-DETR](https://github.com/Traffic-X/ViT-CoMer/tree/Co-DETR).

| Method     | Backbone      | Pretrain                                                                                                                                                                        | Epoch | box AP | mask AP | Config                                                                           | Ckpt | Log                                                                                                                |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------:|
| Co-DETR | ViT-CoMer-L | Beit2<span>*</span>                                                                                                | 16e   | 64.3   | -  | [config](https://github.com/Traffic-X/ViT-CoMer/blob/Co-DETR/projects/configs/co_dino/co_dino_5scale_vitcomer_sfp_16e.py)         | - | - |
| Co-DETR | ViT-CoMer-L | Beit2                                                                                               | 16e   | 62.1   | -  | [config](https://github.com/Traffic-X/ViT-CoMer/blob/Co-DETR/projects/configs/co_dino/co_dino_5scale_vitcomer_sfp_16e.py)         | - | - |
| Co-DINO | Swin-L | ImageNet-22K                                                                                                | 36e   | 60.0   | -  | [config]([./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py](https://github.com/Sense-X/Co-DETR/blob/main/projects/configs/co_dino/co_dino_5scale_swin_large_3x_coco.py)) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing)  | - |

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
