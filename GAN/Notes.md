# Generatrion

生成器generator学习分布，因为不是分布的话，如果和之前的话，Network的输出可能会存在歧义的情况；比如说下图中的情况，对视频进行预测，根据前面的帧来进行预测的话，下一帧可能会向左转，或者向右转，于是视频中就出现了分裂的情况；

![distribution](figures/Snipaste_2022-05-22_15-30-16.png)

因此这种学习分布的Generator适合于需要创造性的任务；

## Generative Adversarial Network(GAN)

GAN网络很多变种，GitHub链接：[the-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)

GAN的基本思路就是一个对抗相互学习的过程；

![basic idea of GAN](figures/Snipaste_2022-05-22_15-41-51.png)

## Theory behind GAN

以 Unconditional GAN 为例，GAN的目标是希望生成数据的分布 P_G 和真实数据分布 P_{data} 尽可能地接近，于是就出现了该怎么样衡量两个分布之间的相似性，divergence 散度；

可以从两个分布中进行采样，计算采样出来的样本之间的相似性；

![gan theory](figures/Snipaste_2022-05-22_15-44-51.png)
![sampling](figures/Snipaste_2022-05-22_15-48-33.png)

对于判别器(Discriminator)而言，旨在训练一个分类器，它能够最大化D、G之间的散度，这里并不是严格意义上的散度，而是用V表示的一个function；

![dicsriminator](figures/Snipaste_2022-05-22_15-59-01.png)

总的来说，GAN的步骤就是：

![gan step](figures/Snipaste_2022-05-22_16-00-03.png)

其他的散度也可以使用；

## Tips for GAN

JS divergence 不太适合；

大部分情况下，P_G 和 P_{data} 并不重叠，这样计算出的JS永远为log2，那这个loss值就没有任何的意义，因为无论generator生成的图片是什么样的效果，discriminator都可以达到百分百的准确率；

![JS diverence](figures/Snipaste_2022-05-22_16-37-09.png)
![JS duverence problem](figures/Snipaste_2022-05-22_16-38-55.png)

于是出现了Wassersteion distance，考虑到两个分布之间的距离，Wassersteion计算从一个分布到另一个分布需要移动的距离，相较于JS divergence有很大的提高；

![problem](figures/Snipaste_2022-05-22_16-51-37.png)

采用了Wassersteion distance的网络当年就是WGAN，公式就是下图中的；重点在于对D有一个限制条件，对这个限制条件后面也有一系列的研究；

![WGAN](figures/Snipaste_2022-05-22_17-00-38.png)
![WGAN condition](figures/Snipaste_2022-05-22_17-02-13.png)

## Conditinal Generation

对于图中的文字转图片的应用来说，这个眼睛颜色就是输入的条件，generation需要根据这个condition决定后面输出的结果；但是对于discriminator来说，不能仅仅判断图像是生成的还是真实的，还需要加上condition

![Conditional GAN](figures/Snipaste_2022-05-22_17-08-06.png)
![Discriminator](figures/Snipaste_2022-05-22_17-58-44.png)
![Discriminator](figures/Snipaste_2022-05-22_17-58-58.png)

## Learning from Unpaired Data

这部分主要涉及到的就是风格迁移，将图像从一个domain转换到另一个domain上；

![style transform](figures/Snipaste_2022-05-22_18-28-28.png)

CycleGAN：

由于没有配对样本，看似可以直接通过generator生成另一个domain里的图片，但是可能生成的跟原始图像之间没有关系，就骗过了discriminator；于是CycleGAN中采用了两个生成器，表示可以将生成的图像又可以转换回原domain中；包含双向的过i成；

![cyclegan](figures/Snipaste_2022-05-22_18-30-26.png)
![cyclegan](figures/Snipaste_2022-05-22_18-30-32.png)
![cyclegan](figures/Snipaste_2022-05-22_18-30-41.png)
![cyclegan](figures/Snipaste_2022-05-22_18-30-48.png)
![cyclegan](figures/Snipaste_2022-05-22_18-31-01.png)

## Evaluation of Generation

人眼评价；
对生成的图像进行分类，结果集中的话就说明这图生成得很好，和真实的类似；

多样性评价：

GAN训练过程中可能会出现生成的图像全是同一类的，这样虽然质量很好，骗过discriminator；Mode Collapse问题；

![mode collapse](figures/Snipaste_2022-05-22_18-44-21.png)

在人脸生成过程中，可能存在generator不断迭代，但是人脸只是肤色变化了的问题，Mode Dropping的问题；

![mode dropping](figures/Snipaste_2022-05-22_18-44-27.png)
