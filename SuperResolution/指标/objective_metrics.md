# 关于指标的欺骗方式

本笔记翻译于 [Ways of cheating on popular objective metrics: blurring, noise, super-resolution and others](https://videoprocessing.ai/metrics/ways-of-cheating-on-popular-objective-metrics.html)

## 引言

在超分辨率论文中，PSNR始终是估计SR图像的最流行标准，始终占主导地位。

![论文指标统计](figures/pic2.png)

PSNR指标的流行与其特征有关：
* 易于计算，指标的运行时间是许多实际应用程序的关键因素；
* 使用历史悠久，经历了长时间实验的验证。很容易将新算法的性能与仅使用PSNR评估的旧算法进行比较；
* 具有明确的物理含义；
* 在数学含义上具有方便的优化方便；

## 著名的批评例子

下图中对原始图像进行不同类型的失真，包括对比度、压缩、模糊、噪声，并且从图像中可以发现改变后的图像具有明显不同的1视觉质量。但是，失真图像的MSE分数是相同的。

![指标批判例子](figures/pic3.png)

另一个例子是，MSE分数几乎相同，但正确图像的视觉质量要好很多，SSIM指标更真实地处理了这一点。

![指标批判例子2](figures/pic4.png)

发生这种情况的原因在于SSIM能够衡量结构相似性，而MSE则没有。这是SSIM与PSNR的真正区别。人类视觉系统对扭曲高度敏感。

然后，SSIM
