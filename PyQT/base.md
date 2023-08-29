# PyQT使用

目前PyQT的版本已经有PyQT5更新成PyQT6；

## 环境准备

安装语句：

```sh
conda create -n develop python=3.9
conda activate develop
pip install sip
pip install PyQt6
pip install PyQt6-tools
```

安装完之后输入下述语句将会打开qtdesigner，这里可以拖动工具设计界面，后缀名是.ui文件；

```sh
pyqt6-tools designer
```

保存ui文件之后，需要将ui文件转换成py代码；

```sh
pyuic6 -x onestopcd.ui -o onestopcd_ui.py
```

* -x 参数后是ui文件；
* -o 参数后是待保存的py文件；

## 代码

## 美化

可搜索pyqt美化，没有更深研究；
