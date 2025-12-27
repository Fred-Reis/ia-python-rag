"""Module providing a RAG pesrsited vector DB."""

import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PERSIST_DIRECTORY = "db"
embedding = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding,
    collection_name="laptop_manual",
)

retriever = vector_store.as_retriever()

SYSTEM_PROMPT = """"
Use o contexto para responder as perguntas
Contexto: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

QUERY = "Qual a marca e modelo do notebook?"

rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke(QUERY)
print(response)
