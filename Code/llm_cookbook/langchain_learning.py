import os
import pandas as pd
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA  #检索QA链，在文档上进行检索
from langchain_community.document_loaders import CSVLoader #文档加载器，采用csv格式存储
from langchain_community.vectorstores import DocArrayInMemorySearch  #向量存储
from langchain.indexes.vectorstore import VectorstoreIndexCreator 
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


if __name__ == '__main__':

    file = r'.\OutdoorClothingCatalog_1000.csv'

    # 使用langchain文档加载器对数据进行导入
    loader = CSVLoader(file_path=file, encoding='utf-8')

    # 使用pandas导入数据，用以查看
    data = pd.read_csv(file,usecols=[1, 2])
    print(data.head())

    # 创建指定向量存储类, 创建完成后，从加载器中调用, 通过文档加载器列表加载
    model_name = "F:/代码库/模型库/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    index = VectorstoreIndexCreator(
        embedding=embeddings,
        vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
    
    llm = ChatTongyi(model='qwen-plus', temperature=0.0)

    query ="请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"

    #使用索引查询创建一个响应，并传入这个查询
    response = index.query(query)

    # llm = ChatTongyi(model='qwen-plus', temperature=0.0)

    # customer_response = llm.invoke(user_message)
    # print(customer_response)
    # print(customer_response.content)
