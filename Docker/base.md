# Docker

## 安装步骤

Windows系统：

Docker Desktop下载网站：[Docker Desktop](https://www.docker.com/products/docker-desktop/)

直接安装即可，安装完后会自动重启电脑，自己注意；

一般可以选择一个基础镜像，然后在该镜像的基础上，将自己的代码、环境什么的都打包进去；
完成之后，可以将自己的镜像上传到阿里云镜像服务上，这样子别人就能下载这个镜像了；

```sh

docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel  # 拉取torch的官方镜像

docker run -it --gpus all pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel  # 启动镜像，并进容器内部

docker build -t wuzm_demo:v1.0.0 .  # 创建镜像，名称：版本号

docker run --gpus all  --network none --rm -it --shm-size 8G -v F:\比赛\CD\fusai\data:/data -v F:\比赛\CD\fusai\test:/test -v F:\比赛\CD\fusai\output:/output wuzm_demo:v1.0.0  # 启动镜像

docker run --gpus all  --network none --rm -it --shm-size 8G -v F:\比赛\CD\fusai\data:/data -v F:\比赛\CD\fusai\test:/test -v F:\比赛\CD\fusai\output:/output wuzm_demo:v1.0.0 bash /workspace/run.sh  # 启动镜像，直接运行sh文件

docker rmi ***  # 删除镜像

docker rm ***  # 删除容器

```

Dockerfile示例：

```sh
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.163.com\/ubuntu\//g' /etc/apt/sources.list
RUN apt update
RUN apt -y upgrade
RUN apt install -y gcc-9
RUN apt install -y g++-9
RUN apt install -y libstdc++6
RUN apt install -y vim
RUN apt-get install zip unzip

# install conda
RUN apt install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.0-3-Linux-x86_64.sh
RUN bash Miniconda3-py38_23.5.0-3-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-py38_23.5.0-3-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init

# install envs
RUN conda install -y -c omgarcia gcc-6
RUN conda install -y libgcc
ENV LD_LIBRARY_PATH=/miniconda/lib/:$LD_LIBRARY_PATH
RUN conda install -y pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
RUN conda install -y gdal -c conda-forge
RUN conda update poppler -y -c conda-forge
RUN pip install albumentations fire tqdm ipdb timm geojson
RUN pip install tensorboard pytorch_tabnet ttach yimage
RUN pip install pyproj addict yapf pyshp shapely
RUN pip install einops fvcore pycocotools matplotlib
RUN pip install prettytable
RUN pip install setuptools==59.5.0
RUN pip install monai

COPY ./code /workspace/code
COPY ./DCNv3-1.0+cu113torch1.12.0-cp38-cp38-linux_x86_64.whl /workspace/
COPY ./detectron2 /workspace/detectron2
COPY ./run.sh /workspace
WORKDIR /workspace/
RUN pip install ./DCNv3-1.0+cu113torch1.12.0-cp38-cp38-linux_x86_64.whl
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN apt-get install -y libgl1

```

## 上传阿里云

阿里云镜像服务： [aliyun](https://cr.console.aliyun.com/cn-hangzhou/instances)
这里是阿里云账户

点击个人实例，镜像仓库，就能看到自己的仓库了，点击管理，下面就会显示一系列命令操作了；

```sh
docker login --username=雾隐之月 registry.cn-hangzhou.aliyuncs.com  # 这里的密码，可以不用与自己的阿里云账户一样
```

密码可自行在个人实例，访问凭证中设定固定密码；
