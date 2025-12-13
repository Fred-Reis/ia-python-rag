import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PDF_PATH = "laptop_manual.pdf"
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents=docs)

embedding = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks, embedding=embedding, collection_name="laptop_manual"
)

retriever = vector_store.as_retriever()

TEMPLATE = """'
"You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
\nQuestion: {question} \nContext: {context} \nAnswer:")
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=TEMPLATE,
)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


try:
    while True:
        question = input("o que vc quer saber do laptop?\n")
        response = rag_chain.invoke(question)
        print(response)

except KeyboardInterrupt:
    exit()
