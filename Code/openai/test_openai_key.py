import openai
    # openai.api_key = 开发者快速获取秘钥参考/uiuiapi。com
#openai.api_key = 'sk-xxxxxx'
    # openai.base_url = url
   # openai.base_url = 'https://api.uiuiapi.com/v1/'
def validate_openai_api_key(api_key):
    """
    验证 OpenAI API Key 是否有效
    """
    openai.api_key = api_key
    try:
        # 使用 ChatGPT 模型测试请求
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="这是一个测试请求，用于验证 API Key 的可用性。",
            max_tokens=5
        )
        print("API Key 验证成功，返回结果：", response.choices[0].text.strip())
    except openai.error.AuthenticationError:
        print("API Key 无效或权限不足，请检查您的 API Key。")
    except Exception as e:
        print("请求失败，错误信息：", e)

# 测试 API Key
api_key = "sk-XXXXXX1234567890abcdefGHIJKLmnopqr"  # 请替换为您的 API Key
validate_openai_api_key(api_key)
