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