# Python 问题

python torch tensorboard  AttributeError: module 'distutils' has no attribute 'version'

去torch tensorboard \_\_init\_\_.py文件中将几行给注释掉：

LooseVersion一行
if两行
del LooseVersion一行

gcc编译DCNv3的时候，出现libmpfr.so.6 cannot open shared object file: No such file or directory
解决方法：

```sh
sudo ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.4 /usr/lib/x86_64-linux-gnu/libmpfr.so.6
```

前面那个目录是链接到的地址，后面那个是需要链接的；
如果提示6的那个文件已存在，

```sh
sudo rm -rf /usr/lib/x86_64-linux-gnu/libmpfr.so.6
```

```python
import cv2
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

apt install libgl1-mesa-glx

libgthread-2.0.so.0: cannot open shared object file: No such file or directory

apt-get install libglib2.0-0
```

问题描述：
gdal属性字段中文乱码，修改编码为GBK没用；有时出现 One or several characters couldn't be converted correctly from GBK to UTF-8.  This warning will not be emitted anymore

解决方法：

记得注册驱动；

```python
gdal.SetConfigOption("SHAPE_ENCODING", "")  # 不设置的话会自动识别是UTF-8还是GBK；
ogr.RegisterAll() # 记得注册驱动
```

问题描述：
version `GLIBCXX_3.4.29‘ not found
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX 结果只到GLIBCXX_3.4.25

解决方法：

```sh
# 查找libstd
sudo find / -name "libstdc++.so.6*"

# 挨个尝试里面能到的最高版本：
strings /home/wuye/anaconda3/envs/tf2/lib/libstdc++.so.6.0.29 | grep GLIBCXX

# 找到一个能行的之后:
sudo cp /home/wuye/anaconda3/envs/tf2/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
# 删除之前链接
sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# 创建新的链接
sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

```
