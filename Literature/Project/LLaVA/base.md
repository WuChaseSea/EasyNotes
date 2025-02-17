# LLaVA: Large Language and Vision Assistant
<!-- omit in toc -->

文章链接：[Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)

## 研究背景与动机

在当前的视觉模型工作中，每一个任务都通过调用单一的大型视觉模型进行解决，任务指引在模型设计中隐式地考虑。语言仅仅用于描述图像内容。然而语言在将图像视觉信号和语言语义信息匹配的过程中具有重要的作用。大语言模型（LLM）展示了语言作为通用接口的潜力，能够通过明确的指令引导模型完成任务。

在本论文中，提出了视觉指令调整(visual instruction-tuning)，第一次尝试将指令调整应用于图像-语言多模态空间中，构建通用视觉助手。

贡献点：

1. 多模态指令跟随数据。当前的一个主要挑战是缺乏视觉-语言指令跟随数据，提出了一种数据重组的方式，使用ChatGPT/GPT-4将图像-文本的形式转换为合适的指令跟随格式。
2. 大型多模态模型。构建了一个大型多模态模型 (large multimodal model, LMM)，连接CLIP的视觉编码器和Vicuna的语言解码器，并在自主构建的指令跟随数据集上进行微调训练。实验证明了使用生成数据用于LMM指令微调的有效性。并提出了构建通用指令跟随视觉代理的实用技巧。当嵌入到GPT4中在Science QA多模态推理数据集上取得了SoTA。
3. 多模态指令跟随的benchmark。为LLaVA-Bench提供了两个基准测试，包括多种成对图像、指令和详细注释。
4. 开源。生成的多模态指令数据集、代码库、模型检查点和视觉聊天演示。

文章链接：[LLaVA: Large Language and Vision Assistant](https://arxiv.org/pdf/2304.08485)
代码链接：[LLaVA](https://github.com/haotian-liu/LLaVA)

## 模型结构

模型的主要目的是联合有效使用预训练的LLM和视觉模型。使用Vicuna作为大语言模型，因为它具有最好的指令跟随能力。

![network architecture](./figures/截屏2025-02-17%2019.34.00.png)

对于一张输入图像，使用预训练的CLIP ViT-L/14的视觉编码器提取特征，使用一个简单的线性层将图像特征转换至单词嵌入空间。

## 目的
