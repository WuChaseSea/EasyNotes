# Docker问题

问题描述：
在容器内部配置好环境之后，docker commit将其保存为镜像，然后将镜像应用至别的电脑上，进入容器之后，cuda nvidia-smi不能正常使用；执行nvidia-smi命令之后，会出现NVIDIA-SMI couldn't find libnvidia-smi.so library in your system. Please make sure that the NVIDIA Display .......，在python窗口中torch也不能调用cuda，torch.cuda.is_available()为False，直接运行深度学习预测程序torch会报错，RuntimeError: Found no NVIDIA driver on your system. Please check........。同时ldconfig命令会显示很多相关的文件是empty；
尝试了网上的一些ln的相关命令之后，nvidia-smi命令能够正常显示显卡信息，但是torch还是不能调用cuda；

原因：
在容器内部配置环境的时候是调用了宿主机的GPU，即在docker run的时候添加了--gpus参数，这样子在执行docker commit的时候会将nvidia driver也打包进行。那么换到另一台电脑上的时候，同样以--gpus的方式启动的话，就会出现文件冲突的问题，导致nvidia不能正常调用。

解决方法：
以普通方式进入容器，手动删除或改名文件/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1和文件/usr/lib/x86_64-linux-gnu/libcuda.so.1 ，然后把此时的容器打包为镜像

```sh
docker run -it --rm id /bin/bash
mv /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1.bak

mv /usr/lib/x86_64-linux-gnu/libcuda.so.1  /usr/lib/x86_64-linux-gnu/libcuda.so.1.bak

# 再开另外一个窗口commit，exit退出去之后容器就删除了
docker commit 
```

问题描述：
在容器配置好环境之后，进入容器内部，apt install命令无法正常运行，报错信息如下：

```sh
apt install git

Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  git-man krb5-locales less libbrotli1 libcbor0.6 libcurl3-gnutls liberror-perl libfido2-1 libgssapi-krb5-2 libk5crypto3 libkeyutils1 libkrb5-3 libkrb5support0 libnghttp2-14
  libpsl5 librtmp1 libssh-4 libxmuu1 openssh-client publicsuffix xauth
Suggested packages:
  gettext-base git-daemon-run | git-daemon-sysvinit git-doc git-el git-email git-gui gitk gitweb git-cvs git-mediawiki git-svn krb5-doc krb5-user keychain libpam-ssh
  monkeysphere ssh-askpass
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
/usr/lib/apt/methods/http: symbol lookup error: /usr/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
E: Method http has died unexpectedly!
E: Sub-process http returned an error code (127)
E: Method /usr/lib/apt/methods/http did not start correctly
```

解决方法：

```sh
conda install libffi==3.3
```

问题描述：
进入容器内部之后，torch不能正常调用cuda，RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environme...

解决方法:

```sh
# 容器内部：
apt install nvidia-modprobe
# 然后重启整台电脑即可
```
