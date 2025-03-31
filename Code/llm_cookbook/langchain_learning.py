import os
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


if __name__ == '__main__':

    # 首先，构造一个提示模版字符串：`template_string`

    template_string = """把由三个反引号分隔的文本\
    翻译成一种{style}风格。\
    文本: ```{text}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    print("\n", prompt_template.messages[0].prompt)

    # llm = Tongyi(model='qwen-plus')
    # print(Tongyi().invoke("你是谁？"))
    customer_style = """正式普通话 \
    用一个平静、尊敬的语气
    """

    customer_email = """
    嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
    更糟糕的是，保修条款可不包括清理我厨房的费用。
    伙计，赶紧给我过来！
    """

    # 使用提示模版
    customer_messages = prompt_template.format_messages(
                        style=customer_style,
                        text=customer_email)
    user_message = [
        SystemMessage(content='你是一个友好的助手'),
        customer_messages[0]
    ]

    llm = ChatTongyi(model='qwen-plus', temperature=0.0)

    customer_response = llm.invoke(user_message)
    print(customer_response)
    print(customer_response.content)
