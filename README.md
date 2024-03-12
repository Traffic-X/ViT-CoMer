## ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions

:fire::fire:The official implementation of the CVPR2024 paper "[ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions](todo)"

:fire::fire:| [Paper](https://arxiv.org/abs/2205.08534) | [ViT-CoMer知乎解读](https://zhuanlan.zhihu.com) | [检测排名图paperwithcode]() |[分割排名图paperwithcode]()|

![](https://img-blog.csdnimg.cn/direct/c4e291f7eb14433087c77009b35c36ca.png#pic_center)


## Highlights


 - We propose a novel dense prediction backbone by combining the plain ViT with CNN features. It effectively `leverages various open-source pre-trained ViT weights`  and incorporates spatial pyramid convolutional features that address the lack of interaction among local ViT features and the challenge of single-scale representation.
 - ViT-CoMer-L achieves **`64.3% AP`** on COCO val2017 without extra training data, and **`62.1% mIoU`** on ADE20K val.



## Introduction
We present a plain, pre-training-free, and feature-enhanced ViT backbone with Convolutional Multi-scale feature interaction, named ViT-CoMer, which facilitates bidirectional interaction between CNN and transformer. Compared to the state-of-the-art, ViT-CoMer has the following advantages: (1) We inject spatial pyramid multi-receptive field convolutional features into the ViT architecture, which effectively alleviates the problems of limited local information interaction and single-feature representation in ViT. (2) We propose a simple and efficient CNN-Transformer bidirectional fusion interaction module that performs multi-scale fusion across hierarchical features, which is beneficial for handling dense prediction tasks. 
![在这里插入图片描述]()
