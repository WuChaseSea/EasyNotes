# SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models

<!-- omit in toc -->

文章链接：[SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models](https://arxiv.org/pdf/2104.14951v2.pdf)

贡献点：

提出一种先进的单图像超分辨率扩散概率模型解决过平滑、模式崩溃的问题。具体来说：

1. SRDiff采用一个预训练的低分辨率encoder，将低分辨率影像转换到隐藏空间；
2. SRDiff采用条件噪声预测器迭代恢复 $x_0$，生成以LR图像为条件的HR图像；
3. 引入残差连接；

