## ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions

:fire::fire:**[CVPR 2024]** The official implementation of the paper "[ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions](https://arxiv.org/abs/2403.07392)"

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-comer-vision-transformer-with/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=vit-comer-vision-transformer-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit-comer-vision-transformer-with/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=vit-comer-vision-transformer-with)

:fire::fire:| [Paper](https://arxiv.org/abs/2403.07392) | [ViT-CoMer知乎解读](https://zhuanlan.zhihu.com/p/687695841) | [ViT-CoMer第三方微信公众号解读](https://mp.weixin.qq.com/s/RHV18ShP15Kq1DSbsiDiKw)

 <div align=center> <img title='vit-comer' src="img/vit_comer.jpg" width = 95% > </div>

**The overall architecture of ViT-CoMer.** ViT-CoMer is a two-branch architecture consisting of three components: (a) a plain ViT with L layers, which is evenly divided into N stages for feature interaction. (b) a CNN branch that employs the proposed Multi-Receptive Field Feature Pyramid (MRFP) module to provide multi-scale spatial features, and (c) a simple and efficient CNN- Transformer Bidirectional Fusion Interaction (CTI) module to integrate the features of the two branches at different stages, enhancing semantic information.


## Highlights

 - We propose a novel dense prediction backbone by combining the plain ViT with CNN features. It effectively `leverages various open-source pre-trained ViT weights`  and incorporates spatial pyramid convolutional features that address the lack of interaction among local ViT features and the challenge of single-scale representation.
 - ViT-CoMer-L achieves **`SOTA 64.3% AP`** on COCO val2017 without training on extra detection data , and **`62.1% mIoU`** on ADE20K val.



## Introduction
We present a plain, pre-training-free, and feature-enhanced ViT backbone with Convolutional Multi-scale feature interaction, named ViT-CoMer, which facilitates bidirectional interaction between CNN and transformer. Compared to the state-of-the-art, ViT-CoMer has the following advantages: (1) We inject spatial pyramid multi-receptive field convolutional features into the ViT architecture, which effectively alleviates the problems of limited local information interaction and single-feature representation in ViT. (2) We propose a simple and efficient CNN-Transformer bidirectional fusion interaction module that performs multi-scale fusion across hierarchical features, which is beneficial for handling dense prediction tasks.
<div align=center>  <img src="img/vit-comer-0.jpg" width = 45%> </div>

## Main Results

**Comparisons with different backbones and frameworks.**  It can be seen that under similar model sizes, ViT-CoMer outper- forms other backbones in the two typical dense prediction tasks of COCO object detection and instance segmentation.
<div align=center>  <img src="img/exp_0.jpg" width = 70% > </div>

**Comparisons with state-of-the-arts.** We conduct experiments based on Co-DETR, using ViT-CoMer as the backbone, and initializing the model with multi-modal pre-training BEiTv2. As shown in Table 4, Our approach outperforms the existing SOTA algorithms without extra training data on COCO val2017, which strongly demonstrates the effectiveness of ViT-CoMer.

<div align=center>
<img src="img/sota.jpg" width = 50% >
</div>

For segmentation, we conduct experiments based on Mask2Former using ViT-CoMer as the backbone, and initializing the model with multi-modal pre-training BEiTv2. As shown in Table 7, our method achieves com- parable performance to SOTA methods on ADE20K with fewer parameters. 

<div align=center>
<img src="img/seg_sota.jpg" width = 50% >
</div>

## News

- [x] ***[20240313]***  create repo
- [x] ***[20240313]***  we release detection code and pretrain weights
- [x] ***[20240315]***  we release ViT-CoMer-L with Co-DETR head configs, which achieves **`64.3 AP`** on COCO 2017val
- [x] ***[20240318]***  we release segementation code and pretrain weights
- [ ] release SOTA weights


## Quick Start
* [**detection**](./detection)
* [**segmentation**](./segmentation)

## Citation

If you find ViT-CoMer useful in your research, please consider giving a star ⭐ and citing:

```
@article{xia2024vit,
  title={ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions},
  author={Xia, Chunlong and Wang, Xinliang and Lv, Feng and Hao, Xin and Shi, Yifeng},
  journal={arXiv preprint arXiv:2403.07392},
  year={2024}
}
```

## Acknowledgements
Many thanks to following codes that help us a lot in building this codebase:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter)


## Contact
If you have any questions while using ViT-CoMer or would like to further discuss implementation details with us, please `leave a message on issues` or contact us directly via email: `xiachunlong@baidu.com`. We will reply as soon as possible.

