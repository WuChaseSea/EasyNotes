# Paddle

paddle-gpu安装（仅限联网情况下）：

```sh
python -m pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta0/windows/windows-cpu-avx-openblas-vs2017/paddlepaddle-3.0.0b1-cp38-cp38-win_amd64.whl
```

参考链接：[paddle download](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html)

检测是否能够正常运行：

```sh
python
import paddle
paddle.utils.run_check()

python -c "import paddle; print(paddle.__version__)"
```

安装PaddleDetection

```sh
git clone https://github.com/PaddlePaddle/PaddleDetection.git
# 安装其他依赖
cd PaddleDetection
pip install -r requirements.txt

# 编译安装paddledet
python setup.py install

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# 检验
python ppdet/modeling/tests/test_architectures.py

# 通过后显示
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK

# 示例：
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg

Downloading simfang.ttf from https://paddledet.bj.bcebos.com/simfang.ttf

```

出现错误：

Could not locate zlibwapi.dll. Please make sure it is in your library path!

解决方法：

```sh
# Windows
找到 zlib.dll文件
C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll
2.复制该文件，并重命名为zlibwapi.dll
3.放到如下路径(记得将该路径添加到环境变量)
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll
4.重新打开终端在执行之前命令，便可成功。

# Ubuntu
sudo apt-get install zlib1g
```
