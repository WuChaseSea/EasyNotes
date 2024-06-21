# CLIP工作改进

<!-- omit in toc -->

CLIP自从2021年提出之后，就应用到方方面面中。在分割领域中就有LSeg、GroupVit，在目标检测领域有ViLD、VLIP V1、V2，在视频中也有video clip、Clip4clip、aciton Clip，和深度图有关的depthClip、pointClip。图像生成领域、多模态领域等都有相关的改进工作。

## CLIP回顾

CLIP以对比学习的方式，训练视觉语言的多模态模型。给定文本图像对，分别以文本编码器、图像编码器进行编码，采用对比损失函数进行训练。在进行zero shot应用时，给定一个template采用文本编码器进行编码，然后与图像特征进行相似度计算，就可以实现图像的分类了。

## 分割

分割和图像分类任务很相似，很多在图像分类上应用的方法都能直接用到图像分割领域。

LANGUAGE-DRIVEN SEMANTIC SEGMENTATION

模型总览图：

有监督语义分割，和普通的语义分割思路很像，进一步引入了文本特征，使得在做推理的时候可以通过文本进行控制，引入文本的信息。训练的时候，通过作者团队之前提出的结构进行特征编码，得到$h \times w \times c$的特征向量，文本分支通过文本编码器得到特征$n \times c$，与图像特征进行相乘之后得到$h \times w \times n$的特征向量，然后decode得到groundtruth，使用cross entropy loss进行优化。论文意义在于给传统的语义分割任务添加了一个文本分支，在训练的时候文本编码器使用clip的文本编码器，并且参数进行了锁定。在推理的时候就可以使用文本进行prompt。
（PS：数据集比较小的情况下，使用基础模型的预训练模型，参数固定会比较好。）

缺点：

相当于还是一种有监督训练的框架，虽然文章采用了7个数据集进行训练，但是总体量也就20W左右的样本量，zero shot能力还是有待改进。

GroupViT: Semantic Segmentation Emerges from Text Supervision

在已有的ViT框架中加入grouping block和group token。目标函数和clip的保持一致，所以zero shot的性能会比较好。训练样本的形式是每一张图都有一句话进行描述。在推理的时候，group token的数量会限制最终结果的预测类别数量。

## 目标检测

OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION

将clip当成teacher，进行蒸馏。

针对现有的目标检测类别单一的问题，提出在现有的数据集上给模型添加开集预测的能力。自己的backbone和clip提取的特征进行一个损失计算。zero shot应用的时候也是通过计算相似度实现的。

Grounded Language-Image Pre-training

vision grounding，根据文本从图像中将物体检测出来。GLIP是一种有监督的训练。

## 其余

CLIPasso: Semantically-Aware Object Sketching

有监督训练，用clip对影像和结果分别进行编码，计算损失值；

CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval

视频理解。一个文本特征、一个影像特征，计算相似度。

ActionCLIP: A New Paradigm for Video Action Recognition

动作识别。使用CLIP摆脱带标签的数据；计算相似度。

How Much Can CLIP Benefit Vision-and-Language Tasks?

将clip用于vision-language之中。将clip用作预训练参数用于各种下游任务上，证明了有效果；

AudioCLIP: Extending CLIP to Image, Text and Audio

语音分类

PointCLIP: Point Cloud Understanding by CLIP

3D点云

Can Language Understand Depth?

深度图生成

## 总结

对clip对模型大致分为3点：

1）将图像或者文本通过clip模型得到一个预训练特征，将这个特征与原来的特征进行一个融合；

2）将clip模型当成一个teacher，蒸馏；

3）借鉴clip多模态对比学习的思想，应用到新任务上，自行定义正样本对、负样本对，计算多模态对比学习loss，从而实现zero shot的detection或者segmentation；

只改动一点点的参数，应用到下游任务上来会是一种更倾向到方法。
