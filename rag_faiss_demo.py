import os
import bs4
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDER = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class Indexer:
    _path = '/index'

    def __init__(self):
        self._text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100)
        self._embedder = EMBEDDER

    def _load_doc(self):
        pdf_docs = self._get_pdf_docs()
        md_docs = self._get_md_docs()
        web_docs = self._get_web_docs()
        return pdf_docs + md_docs + web_docs


    @classmethod
    def _get_pdf_docs(cls):
        loader = DirectoryLoader(
            "data/",
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            # loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        return docs

    @classmethod
    def _get_md_docs(cls):
        loader = DirectoryLoader(
            "data/",
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        return docs

    @classmethod
    def _get_web_docs(cls):
        loader = WebBaseLoader(
            web_paths=("https://deep-purple.ru/history/dp_bio.html",),
            # bs_kwargs={
            #     "parse_only": bs4.SoupStrainer(id="content")
            # }
        )
        docs = loader.load()
        return docs

    def create_index(self):
        os.makedirs('index', exist_ok=True)
        docs = self._load_doc()
        split_docs = self._text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(split_docs, self._embedder)
        vector_store.save_local("index/my_faiss_index")

class Retriever:
    def __init__(self):
        self._embedder = EMBEDDER
        self._vector_store = FAISS.load_local(
            "index/my_faiss_index",
            self._embedder,
            allow_dangerous_deserialization=True
        )

    def extract(self, query: str):
        retriever = self._vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        return relevant_docs


if __name__ == "__main__":
    # Indexer().create_index()
    extractor = Retriever()
    for doc in extractor.extract('Don Airey on keyboards'):
        print(doc)