# TensorRT加速使用

## 环境配置

下载网址： [https://developer.nvidia.cn/tensorrt](https://developer.nvidia.cn/tensorrt)
需要登陆之后下载，选择合适的版本，TAR压缩包或者deb都可，这里选择的TAR压缩包的形式，deb的还没有试过；
解压缩：

```sh
tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
vim ~/.bashrc
# 添加以下内容
export PATH=/path/to/TensorRT-8.6.1.6/bin:$$PATH
export LD_LIBRARY_PATH=/path/to/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/TensorRT-8.6.1.6/lib::$LIBRARY_PATH
source ~/.bashrc

trtexec # 能输出即可

cd python
pip install whl # 选择合适的whl安装
```

```python
import tensorrt
tensorrt.__version__
```

安装onnxruntime

```sh
pip install onnx
pip install onnxruntime-gpu
```

```python
import onnxruntime as onnx
onnx.get_device()
```

转换成onnx形式；
转换的过程中会出现opset的问题，UnsupportedOperatorError: Exporting the operator 'aten::unflatten' to ONNX is not supported.主要原因是torch版本不对，可以使用最新的或者比较旧的，我是更新至2.1.0的torch；
检查onnx形式；
转换engine形式；
调用engine；
