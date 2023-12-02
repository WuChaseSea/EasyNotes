# Ubuntu环境配置问题

离线安装gcc、g++

正常情况下：

```sh
sudo apt install gcc-7
sudo apt install g++-7
```

apt 和 apt-get的区别在于apt是直接联网安装，apt-get则会将文件下载到文件；

离线情况下，需要下载deb文件，但是由于需要的deb文件较多，有时候下载起来会比较麻烦；

一种方法是：在有网的ubuntu系统上下载好所有的deb文件，然后拷贝到离线机器上，之后运行：

```sh
sudo dpkg -i *.deb
```

以Windows系统为例，在应用商店中安装对应的ubuntu系统版本，我这儿下的是ubuntu18；

进去之后：

```sh
cd /var/cache/apt/archives  # 这里面都是一些下载的缓存文件
sudo apt clean all  # 清空缓存目录，防止其他的deb文件干扰
sudo apt-get update
sudo apt-get install -y gcc  # 下载gcc的文件，并安装，中间出现一些error无所谓
sudo cp -r archives/ /mnt/f/yourfoldername  # 将文件夹复制，子系统的盘在/mnt中
gcc -v
sudo apt clean all
sudo apt-get install -y g++
```

编译文件的时候，出现libmpfr.so.6的问题：

ln命令前面那个是源目标，后面那个是目标文件，比如说我的libmpfr.so.6找不到，那么我就将libmpfr.so.4给libmpfr.so.6的链接；

```sh
sudo ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.4 /usr/lib/x86_64-linux-gnu/libmpfr.so.6
```
