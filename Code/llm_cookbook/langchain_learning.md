# 使用LangChain开发应用程序

## 模型、提示和输出解释器

模型指使用Langchain中定义的模型，提示是使用提示词模板，输出解释器指在输出中指定输出格式；

## 储存

使用 LangChain 中的储存(Memory)模块时，它旨在保存、组织和跟踪整个对话的历史，从而为用户和模型之间的交互提供连续的上下文。

LangChain 提供了多种储存类型。其中，缓冲区储存允许保留最近的聊天消息，摘要储存则提供了对整个对话的摘要。实体储存则允许在多轮对话中保留有关特定实体的信息。这些记忆组件都是模块化的，可与其他组件组合使用，从而增强机器人的对话管理能力。储存模块可以通过简单的 API 调用来访问和更新，允许开发人员更轻松地实现对话历史记录的管理和维护。

对话缓存储存，ConversationBufferMemory对模型的对话以字典的形式进行储存。对话缓存窗口储存ConversationBufferWindowMemory通过参数k的设置保留几个对话记忆次数。对话字符缓存储存ConversationTokenBufferMemory限制保存的Token数量。对话摘要缓存储存ConversationSummaryBufferMemory使用 LLM 对到目前为止历史对话自动总结摘要，并将其保存下来。

```python
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 这里我们将参数temperature设置为0.0，从而减少生成答案的随机性。
# 如果你想要每次得到不一样的有新意的答案，可以尝试增大该参数。
llm = ChatOpenAI(temperature=0.0)  
memory = ConversationBufferMemory()


# 新建一个 ConversationChain Class 实例
# verbose参数设置为True时，程序会输出更详细的信息，以提供更多的调试或运行时信息。
# 相反，当将verbose参数设置为False时，程序会以更简洁的方式运行，只输出关键的信息。
conversation = ConversationChain(llm=llm, memory = memory, verbose=True )
```

## 模型链

链（Chains）通常将大语言模型（LLM）与提示（Prompt）结合在一起，基于此，我们可以对文本或数据进行一系列操作。链（Chains）可以一次性接受多个输入。例如，我们可以创建一个链，该链接受用户输入，使用提示模板对其进行格式化，然后将格式化的响应传递给 LLM 。我们可以通过将多个链组合在一起，或者通过将链与其他组件组合在一起来构建更复杂的链。

### 大语言模型链

大语言模型链LLMChain，将大语言模型(LLM)和提示（Prompt）组合成链。这个大语言模型链非常简单，以一种顺序的方式去通过运行提示并且结合到大语言模型中。

```python
from langchain.chains import LLMChain   

chain = LLMChain(llm=llm, prompt=prompt)

product = "大号床单套装"
chain.run(product)
```

### 简单顺序链

顺序链（SequentialChains）是按预定义顺序执行其链接的链。每个步骤都有一个输入/输出，一个步骤的输出是下一个步骤的输入。

```python
from langchain.chains import SimpleSequentialChain

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True)
```

### 顺序链

当只有一个输入和一个输出时，简单顺序链（SimpleSequentialChain）即可实现。当有多个输入或多个输出时，则需要使用顺序链（SequentialChain）来实现。

```python
from langchain.chains import LLMChain  
from langchain.chains import SequentialChain

#输入：review    
#输出：英文review，总结，后续回复 
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],  # 这个是第一个链的输出
    output_variables=["English_Review", "summary","followup_message"],  # 这个是不同链的输出
    verbose=True
)
```

### 路由链

对于更复杂的事情，根据输入将其路由到一条链，具体取决于该输入到底是什么。如果你有多个子链，每个子链都专门用于特定类型的输入，那么可以组成一个路由链，它首先决定将它传递给哪个子链，然后将它传递给那个链。

路由器由两个组件组成：

* 路由链（Router Chain）：路由器链本身，负责选择要调用的下一个链
* destination_chains：路由器链可以路由到的链

```python
from langchain.chains.router import MultiPromptChain  #导入多提示链
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
```

## 基于文档的问答

使用大语言模型构建一个能够回答关于给定文档和文档集合的问答系统是一种非常实用和有效的应用场景。与仅依赖模型预训练知识不同，这种方法可以进一步整合用户自有数据，实现更加个性化和专业的问答服务。例如,我们可以收集某公司的内部文档、产品说明书等文字资料，导入问答系统中。然后用户针对这些文档提出问题时，系统可以先在文档中检索相关信息，再提供给语言模型生成答案。
