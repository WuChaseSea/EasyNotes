# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
<!-- omit in toc -->

文章链接：[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
代码链接：[BLIP-1](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## 目的

减少计算成本并避免灾难性遗忘，

## 贡献点

* BLIP-2有效地利用了预训练的图像模型和语言模型，在图像和语言之间构建了两阶段预训练的Q-Former：表示学习阶段和生成学习阶段。
* 基于大语言模型驱动，BLIP-2可以实现自然语言引导的零样本图文生成；
* 使用了冻结的单模型和轻量级的Q-Former，BLIP-2实现了最优的计算效率；

## 模型

![model](./figures/Snipaste_2024-10-17_10-48-30.png)

提出了一种多模态混合编码-解码结构(multimodal mixture of encoder-decoder, MED)，一种多任务模型，能够运行任意一种形式：

1. Unimodal encoder, 单独编码图像和文字，文本编码器和BERT类似，添加一个 [CLS] token 在句子前面；
2. Image-ground text encoder，通过交叉注意力机制注入视觉信息，添加 [Encode] token 作为多模态图文对的表示；
3. Image-ground text decoder，利用causal self-attention layers替换之前的bi-directional self attention layers；

通过这种模型设计，计算复杂度较高的图像编码器只用前传一次，文本编码器则前传三次

Image-Text Contrastive Loss(ITC)，对齐视觉图特征和文本特征；
Image-Text Matching Loss(ITM)，在给定图像特征和文本特征的情况下，判断它们是否是匹配的；为了获取更多的负样本，使用了困难负样本挖掘策略，因此在计算loss的时候具有更相反的特征的负例更容易被选中；
Language Modeling Loss(LM)，计算交叉熵损失，解码文本描述；

为了在多任务学习中有效的预训练，文本编码器和文本解码器除了自注意力层，其余层共用同一套参数。原因是因为编码和解码任务中的差异都是通过SA层获取的。encoder使用bi-directional self-attention 构建当前token的特征表示，decoder使用casual self-attention预测下一个token。其余层共享参数能够改善训练效率。

![model](./figures/截屏2024-11-27%2020.10.17.png)
过滤操作；
