# Python常用命令

## 环境安装

```sh
# 创建虚拟环境
conda create -n develop python=3.9
```

## 环境分享

```sh
pip freeze > requirements.txt  # 这样导出的文件中存在自己系统的文件路径
pip list --format=freeze > requirements.txt  # 只包含安装库的版本，不包含文件路径
```

```sh
pip install conda-pack

conda pack -n myenv -o myenv.tar.gz --ignore-missing-files --ignore-editable-packages

mkdir my_env ## 创建文件夹，不然tar解压缩指定目录会报错

tar -xvzf myenv.tar.gz -C my_env
```

## 常用库的安装和使用

* Crypto

```sh
pip install pycryptodome  # Windows
pip install pycrypto  # Linux
```

* opencv

读取图片参数：

```sh
cv2.imread()
-1: cv2.IMREAD_UNCHANGED
0: cv2.IMREAD_GRAYSCALE
1: cv2.IMREAD_COLOR
```

保存图片：

```sh
cv2.imwrite(filepath, imgdata)
```

改变大小：

```sh
cv2.resize(imgdata, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.INTER_NEAREST 最近邻
cv2.INTER_CUBIC 三次
cv2.INTER_LINEAR 默认，线性插值
cv2.INTER_AREA 区域插值
```

**对标签进行重采样时一定要注意插值方法，不要插入新值**

* pathlib

```python
Path('./data').mkdir(parents=True, exist_ok=True)  # 创建目录
.stem  # 文件名
.suffix  # 后缀名
.parent  # 父文件夹路径
```

## 常见情况

jupyter转py

```sh
jupyter nbconvert --to script xxx.ipynb
```
