# MaskFormer

<!-- omit in toc -->

## MaskFormer

MaskFormer中的查询数目就是指潜在的实例对象，与真实值的每一个实例计算损失；
实例分割中，每一个实例对象有class，mask；
二分类的语义分割中（实例分割），每一个实例对象也有class、mask，只不过class从0、1中选择；
一般情况下，查询数目和真实影像中的实例对象数目肯定不相等；
当查询数目小于实例对象数目时，没有问题；
当查询数目大于实例对象数目时，需要加大查询数目，但是这样就会导致占用内存过大，根据GPT的回答，还可以只计算匹配的实例对象，但是感觉不太行；

通过对预测结果的每一个潜在的实例对象打印出来可知，潜在的终归只是潜在的，并不代表它就是预测的图中真正的单个实例对象，它很大概率是一个潜在的值对应很多的实例对象；

## Mask2Former

## MPFormer
