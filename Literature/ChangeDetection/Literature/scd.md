# 普通文献

标题：Joint Spatio-Temporal Modeling for Semantic Change Detection in Remote Sensing Images
联合时空建模的遥感语义变化检测；
贡献点：
1）提出了SCanNet学习SCD中的语义转换，使用三个encoder-decoder学习语义特征和变化特征，SCanFormer对语义特征进行长时序建模；
2）提出了semantic learning scheme，考虑任务的先验知识，分别集入了语义信息的先验和变化信息的先验；

损失值的计算：
1）语义分割的损失，对前后时相的语义标签，计算语义损失；
2）变化损失，将预测的变化检测结果与真实的变化检测计算损失；
3）变化相似度损失，通过变化的标签，在变化区域内前后时相的语义应尽可能相似；

评价指标：
1）accuracy，前后时相的语义预测结果预测正确的像素数目；
2）fscd，变化区域内的语义分割效果；
3）mIoU，计算变化和未变化的IoU；
4）SeK，计算变化区域内的分割效果；
