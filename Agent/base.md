# Agent笔记

仓库：[TinyAgent](https://github.com/KMnO4-zx/TinyAgent/tree/master)
描述：使用agent调用谷歌api接口，使用的模型是internlm2-chat-7b，但是3070只有10GB，带不动；会出现 **RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.For debugging consider passing CUDA_LAUNCH_BLOCKING=1.Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.**

问题1：OSError: Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.
原因：transformers库通过目录导入的时候，目录中不能有中文；

问题2：ModuleNotFoundError: No module named 'transformers.cache_utils'
原因：pip install --upgrade transformers sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple
Successfully installed tokenizers-0.20.0 transformers-4.45.0

目前只能使用 internlm2-chat-1_8b，或者 internlm2_5-1_8b-chat试试
链接：
[internlm2-chat-1_8b](https://huggingface.co/internlm/internlm2-chat-1_8b)
[internlm2_5-1_8b-chat](https://github.com/InternLM/InternLM/tree/main)
