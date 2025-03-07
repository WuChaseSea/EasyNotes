# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

<!-- omit in toc -->

文章链接：[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html)
代码链接：[BLIP](https://github.com/salesforce/BLIP)

## 贡献点

* 编码器-解码器的多模态混合结构MED，用于多任务预训练和灵活的迁移学习的模型结构。它可以是单模态的图像编码器、基于图像的文本编码器和基于图像的文本解码器。使用三个视觉语言目标联合训练，图文对比损失ITC，图文匹配ITM，图像条件语言建模LM
* 标题和过滤（CapFilt），从有噪声的图文数据集中学习的引导方法，captioner对图像合成描述，filter从原始网络文本和合成文本中移除噪声文本。

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
