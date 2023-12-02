# FastSAM

[Code](https://github.com/CASIA-IVA-Lab/FastSAM)

## 为什么会有FastSAM

SAM在很多计算机视觉任务上都有较好的效果，但是SAM在工程应用场景下显得计算量太大，主要是因为中间采用的Transformer结构。SAM主要是一个分割生成和提示的任务，作者认为可以用一个带有实例分割分支的CNN检测器来完成这个任务。

## 贡献点

* 基于CNN结构的实时语义分割任务方法；
* 首次将CNN检测器用于segment anything上，希望将轻量CNN模型用于复杂视觉任务上；
* 在多个细分领域和SAM进行了比较；

## 结构

包含两个分支，检测分支和实例分支；

训练包含两个阶段，第一个是先对整张图片进行所有实例的分割；第二步是根据prompt获取所需区域；
