# Pytorch常见问题

问题：训练阶段可以正常运行，验证阶段出现cuda out of memory的问题

解决方法：在调用模型进行验证或者预测时，加上with torch.no_grad

```sh
with torch.no_grad():  # 加上这句
    outputs = model.forward(data)
```
