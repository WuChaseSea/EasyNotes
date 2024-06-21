# MaskCD: A Remote Sensing Change Detection Network Based on Mask Classification

现有的变化检测网络从逐像素的角度出发，但是这种方法在边界和物体完整性上还有很大的不足。在高分辨率遥感影像中，单个物体的完整性比单个像素更值得关注。因此，本文从掩膜预测和分类的角度重新审视变化检测，提出了MaskCD，自适应生成分类的masks。具体来说，使用跨层变化表征感知者（CLCRP）学习不同尺度的变化检测特征，捕获可形变多头自注意力机制（DeformMHSA）特征之间的时空联系。基于掩膜注意力的transformer解码器（MA-DETR）用于准确地对变化目标进行定位，通过对像素级的特征进行解码获取感兴趣的变化区域。

贡献点：

* 提出一种先进的基于掩膜分类的变化检测模型 MaskCD，以端到端端方式将图像中的变化识别为单个对象；
* CLCRP生成多层次的变化表征，加强Swin Transformer提取特征过程中双时相特征的时空联系；
* 解码器MA-DETR用于定位不同的变化区域，能够抑制背景噪声，检索所有的感兴趣区域；
* 结果达到SoTA；

![network](figures/截屏2024-04-29%2021.50.47.png)

前面使用swin transformer作为backbone提取特征，然后提出一个可形变卷机的模块加强双时相特征之间的联系；decoder采用提出的MA-DETR，基于mask进行自注意力机制；

总体来说，文章表示从mask的角度进行变化检测能够解决逐像素预测出现的物体不连续的问题；
