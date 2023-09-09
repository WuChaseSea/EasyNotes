# 损失函数

## 语义分割

(1) 交叉熵损失函数
实现方式有：

```python
import torch.nn.functional as F
nn.BCELoss(F.sigmoid(input), target)  # 二元交叉熵
nn.CrossEntropyLoss(input, target) # 多分类交叉熵
```

当背景数量，y=0的数量远大于前景像素数量，y=1的数量，损失函数中y=0的成分占主导，模型偏向于背景；

(2) 带权交叉熵

(3) Focal Loss

(4) Dice Loss
dice是一个评价指标，dice loss = 1 - dice metric；
一般情况下使用Dice Loss会有反向传播不利的影响，训练不稳定；


