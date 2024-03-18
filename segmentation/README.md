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

| Method     | Backbone      | Pretrain                                                                                                                                                                        | Lr schd | Crop Size | mIoU(SS/MS) | Config                                                                           | Ckpt | Log                                                                                                                |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------:|
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/1-2a--MV1yVemzM1QX_0bNQ?pwd=r9uv)                                                                                                 | 1×   | 52.1   | 45.8   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1BMb14R4XaTG0wxbWQWoIGQ?pwd=tkc5)  | [log](https://pan.baidu.com/s/1yW7DoMDTdjeSkNQOA2vwzw?pwd=n62v) |
| Mask R-CNN | ViT-CoMer-S | [DINOv2-S](https://pan.baidu.com/s/1-2a--MV1yVemzM1QX_0bNQ?pwd=r9uv)                                                                                                 | 3×   | 48.6   | 42.9   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1nxgjko_7m_I6OQxEGK__IA?pwd=x5a4)  | [log](https://pan.baidu.com/s/1il2nrRRkRIWv_fVycimn0A?pwd=np4p) |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1gjuuFmYl_cNCc8y7ZE5_rg?pwd=5ngw)                                                                                                 | 1×   | 52.0   | 45.5   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_1x_coco.py)         | [ckpt](https://pan.baidu.com/s/1Nqn3QS2jy0wyn-aKBcbGyg?pwd=derg)  | [log](https://pan.baidu.com/s/1-L9XexL1C8vlrJh9J8X_Yg?pwd=qt9a) |
| Mask R-CNN | ViT-CoMer-B | [DINOv2-B](https://pan.baidu.com/s/1gjuuFmYl_cNCc8y7ZE5_rg?pwd=5ngw)                                                                                                 | 3×   | 54.2   | 47.6   | [config](./configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_base_fpn_3x_coco.py)         | [ckpt](https://pan.baidu.com/s/1dUAJ_ToRkNhPrcpqQmgcGQ?pwd=8iam)  | [log](https://pan.baidu.com/s/16byNOInQ1JJ4arjAYtuIMA?pwd=d5ud) |

## Evaluation

To evaluate ViT-CoMer-T + UperNet (512) on ADE20k val on a single node with 8 gpus run:

```shell
sh test.sh
```


## Training

To train ViT-Adapter-T + UperNet on ADE20k on a single node with 8 gpus run:

```shell
sh train.sh
```
