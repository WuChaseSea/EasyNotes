# -*-coding:UTF-8 -*-
'''
* prompt.py
* @author wuzm
* created 2025/03/15 18:37:33
* @function: llm-cookbook prompt python file
'''
import os
from openai import OpenAI


# 一个封装 Qwen 接口的函数，参数为 Prompt，返回对应结果
def get_completion(prompt, model="qwen-plus", temperature=0):
    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 qwen-plus
    '''
    try:
        client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'user', 'content': prompt}
                ],
            temperature=0, # 模型输出的温度系数，控制输出的随机程度
        )
        return completion.choices[0].message.content
    except  Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return 

def get_completion_from_messages(messages, model="qwen-plus", temperature=0):
    try:
        client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            temperature=temperature, # 模型输出的温度系数，控制输出的随机程度
        )
        return completion.choices[0].message.content
    except  Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return 
    

if __name__ == "__main__":
    messages =  [  
    {'role':'system', 'content':'你是个友好的聊天机器人。'},    
    {'role':'user', 'content':'Hi, 我是Isa。'}  ]
    response = get_completion_from_messages(messages, temperature=1)
    print(response)
    