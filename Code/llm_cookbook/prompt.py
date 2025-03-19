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
def get_completion(prompt, model="qwen-plus"):
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
    

if __name__ == "__main__":
    # 不满足条件的输入（text中未提供预期指令）
    text_2 = f"""
    今天阳光明媚，鸟儿在歌唱。\
    这是一个去公园散步的美好日子。\
    鲜花盛开，树枝在微风中轻轻摇曳。\
    人们外出享受着这美好的天气，有些人在野餐，有些人在玩游戏或者在草地上放松。\
    这是一个完美的日子，可以在户外度过并欣赏大自然的美景。
    """
    prompt = f"""
    您将获得由三个引号括起来的文本。\
    如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

    第一步 - ...
    第二步 - …
    …
    第N步 - …

    如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
    \"\"\"{text_2}\"\"\"
    """
    response = get_completion(prompt)
    print("Text 2 的总结:")
    print(response)