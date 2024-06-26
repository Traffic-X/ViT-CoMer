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
| UperNet | ViT-CoMer-T | [DeiT-T](https://pan.baidu.com/s/1684XaK4dRb8crxb8DRrQ7Q?pwd=fxqa)                                                                                                 | 160k   | 512   | 43.5/-   |     38.7M  | [config](https://pan.baidu.com/s/1KxzkLZu8qXi9wfIe3JF04w?pwd=4gjs)  |[ckpt](https://pan.baidu.com/s/1J_XgJ058PpK8gqz9E0Caig?pwd=k6mf)   |[log](https://pan.baidu.com/s/1qh6xvubnU9Y6bG6UNp22IA?pwd=3p8u)|
| UperNet | ViT-CoMer-S | [DeiT-S](https://pan.baidu.com/s/1HCvcilNKPgCp4gYbsSLQpw?pwd=p4jg)                                                                                                 | 160k   | 512   | 46.5/-   |     61.4M  | [config](https://pan.baidu.com/s/1H3PC01bMQvquRLvd4JHuuA?pwd=kgyy)  |[ckpt](https://pan.baidu.com/s/1CDfKeUzCTs5fB0ggy9wYwg?pwd=puqi)   |[log](https://pan.baidu.com/s/1nci50aHO0ma3YgIzH-z9NQ?pwd=cxdj)|
| UperNet | ViT-CoMer-B | [DeiT-S](https://pan.baidu.com/s/1XuTrT95i1XC52bzYeFdIQw?pwd=9kab)                                                                                                 | 160k   | 512   | 48.8/-   |     144.7M  | - |-  |-|


**COCO-Stuff-164K**

| Method  | Backbone   | Pretrain  | Lr schd | Crop Size | mIoU(SS/MS) | #Param  | Config | Ckpt |Log |
|:----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:------:|:-------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:-------------:|:-------------:|
| Mask2Former | ViT-CoMer-L | [BEiTv2-L](https://pan.baidu.com/s/1A4LYOW0uljCnhX_HDXxUdQ?pwd=mxfw)                                                                                                 | 80k   | 896   | 52.7/-   |     633.6M  | [config](https://pan.baidu.com/s/1ooq93rVskLe0kJj3JB5ahA?pwd=ts6k)  |[ckpt](https://pan.baidu.com/s/16fDCV4D8ldNjEUGWhh6WJA?pwd=xnjy)   |[log](https://pan.baidu.com/s/16Zi_ZA8Gi4YtVPzXNr_IrA?pwd=tdfi)|



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
