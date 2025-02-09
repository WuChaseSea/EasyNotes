# LLaVA: Large Language and Vision Assistant
<!-- omit in toc -->

文章链接：[LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2310.03744)
代码链接：[LLaVA](https://github.com/haotian-liu/LLaVA)

## 目的

在本地机器Windows系统上实现LLaVA模型图片描述

## 步骤

1. 下载相关仓库：

```sh
git clone https://github.com/haotian-liu/LLaVA
```

从huggingface上下载相关模型：![LLaVA HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-7b)

2. 进入仓库运行如下命令可以在CMD中对话式：

```sh
python -m llava.serve.cli --model-path ./LLaVA/llava-v1.5-7b --image-file "https://llava-vl.github.io/static/images/view.jpg" --load-4bit
```

3. 运行python程序对单张图进行描述：

```python

from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

model_path = "./LLaVA/llava-v1.5-7b"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output


if __name__ == '__main__':
    sorted_file_names = [
        './images/geosynth_generated_city.jpg'
    ]
    for file_name in sorted_file_names:
        try:
            image, output = caption_image(file_name, 'Describe the contents of the image')
            print(output)
            # image
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

```

## 遇见问题

```sh
importlib.metadata.PackageNotFoundError: bitsandbytes
```

解决方法：

```sh
pip install bitsandbytes

# or download whl files 
# https://pypi.org/project/bitsandbytes/#files

python -m bitsandbytes # run succeed. Installation was successful!

```

```sh
ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time.

pip install transformers==4.37.2

```

```sh
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like openai/clip-vit-large-patch14-336 is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

解决方法：没有下载clip的相关模型。下载clip-vit-large-patch14-336的文件在一个文件夹之后，在之前下载的llava-v1.5-7b的模型文件夹中config.json中修改openai/clip-vit-large-patch14-336的路径为刚刚的路径；
