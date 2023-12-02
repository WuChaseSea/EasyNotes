# SEEM

Code: [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/tree/v1.0)

## 代码运行

下载main分支代码，里面有demo代码；

```sh
python app.py
```

即可；

问题：没有pt文件；
**解决方法**：需要提前在提供的网址上将pt文件下载后放到demo_code文件夹下；
[Hugging Face SEEM](https://huggingface.co/xdecoder/SEEM/tree/main)

问题：缺少clip的相关文件；
**解决方法**：按照打印日志的提示将clip的相关文件下载，放到openai/clip-vit-base-patch32文件夹中；
[Hugging Face openai clip-vit-batch-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)

问题：cannot import name 'distance_transform' from 'kornia.contrib'
**解决方法**：
Windows系统：[Microsoft](https://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/download/details.aspx%3Fid%3D57467)选择下载；
Linux系统：

```sh
sudo apt-get install mpich
```

问题：module 'whisper' has no attribute 'load_model'
**解决方法**：

```sh
import whisper
print(whisper.__file__)  # 会发现为空的

pip uninstall whisper
pip install git+https://github.com/openai/whisper.git 
```

注意whisper的git地址，别下载错了；

问题：gradio __init__() got an unexpected keyword argument 'source' 或者 gradio keyerror dataset
**解决方法**：gradio版本不对，之前根据requirement下载的3.31.0会出现后面的错，就更新为3.37.0

```sh
pip install gradio==3.37.0
```
