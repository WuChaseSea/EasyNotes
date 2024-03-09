# MM系列环境配置

```sh
pip install mmengine (网站上的先安装mim，然后使用mim install之前也试过，不太行)
cd mmsegmemtation-main
python -m pip install -v -e .
cd mmdetection-main
python -m pip install -v -e .
cd mmcv-2.0.1
python -m pip install -e . -v
python .dev_scripts/check_installation.py  # 检查mmcv是否正常
```

mmcv的安装比较麻烦，直接采用提供的whl进行安装2.0以上的版本的话，cuda只能使用10.2的；而我使用了11.6，就从头开始编译了；编译时间会比较长，半小时左右；（半小时好像是卡的问题，一般几分钟就好了）

