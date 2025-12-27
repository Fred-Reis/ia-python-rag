"""Module providing a RAG pesrsited vector DB."""

import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

CSV_PATH = "carros.csv"

loader = CSVLoader(CSV_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(
    documents=docs,
)

PERSIST_DIRECTORY = "db"
embedding = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=PERSIST_DIRECTORY,
    collection_name="cars",
)
