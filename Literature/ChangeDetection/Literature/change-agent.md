# Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis

观测地表方面的变化对于理解人类影响是至关重要的。遥感卫星影像为观测这些变化提供了一个特别的角度，导致了遥感影像变化理解（remote sensing image change interpretation, RSICI）是一个重要的研究重点。现有的RSICI包含变化检测和变化描述两个方向，每一个方向都有一定的限制。为了解决这个问题，提出了一种可交互的Change-Agent，可以根据用户的指令实现综合的变化理解和分析，比如变化检测、变化描述、变化物体统计、变化原因分析等。

贡献点：

* 构建一种多层次的变化理解（multi-level change interpretation）数据集。基于LEVIRCD数据集，添加了变化描述；
* 提出一种双分支MCI模型，提供基于像素层次和语义层次的变化理解信息。除此，提出带有LPE、GDFA的BI3层加强模型的变化理解能力；
* 基于MCI模型和LLM，构建了一个Change-Agent，实现交互的综合理解分析地表变化，具有智能对话和定制话服务能力，为遥感智能应用提供了新思路；

![network](figures/截屏2024-04-30%2016.17.59.png)

文章提出了一个变化理解的数据集，提出了一个多任务训练变化理解模型，然后结合LLM实现智能应用；
