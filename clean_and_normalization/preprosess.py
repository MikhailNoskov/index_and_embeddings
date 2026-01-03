import re
import logging
import argparse
import numpy as np
from typing import List, Callable
from functools import partial

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def apply_func_to_all_docs(func: Callable):
    def process_docs(docs):
        for doc in docs:
            doc.page_content = func(doc.page_content)
        return docs
    return process_docs


@RunnableLambda
def get_md_docs(*args):
    loader = DirectoryLoader(
        "docs/",
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    return docs


@RunnableLambda
def get_pdf_docs(*args):
    loader = DirectoryLoader(
        "docs/",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    return docs


@RunnableLambda
@apply_func_to_all_docs
def clean_text(text: str) -> str:
    # Удаляем упоминания "Company Confidential ..."
    text = re.sub(r"Company\s+Confidential\s+\d{4}\.\s+Все права защищены\.", " ", text)
    # Убираем 'Содержание...' как пример ненужного раздела
    text = re.sub(r"\.{3,}Содержание\.{3,}", " ", text)
    # Удаляем лишние переводы строк (больше 2 подряд превращаем в 1, одиночные в пробел)
    text = re.sub(r"\n{3,}", "\n\n", text)   # больше двух \n -> два \n
    text = re.sub(r"[ \t]+\n", "\n", text)   # убираем пробелы в конце строк
    text = re.sub(r"\n", " ", text)          # заменяем оставшиеся переводы строк пробелом
    # Убираем двойные пробелы
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


@RunnableLambda
def check_and_update_meta(docs: List[Document]) -> List[Document]:
    checked = []
    for doc in docs:
        if doc.metadata:
            doc.metadata.update({"content_len": len(doc.page_content)})
            checked.append(doc)
        else:
            logging.info(f"{doc.metadata['source']} skipped for it does not have metadata")
    return checked


@RunnableLambda
@apply_func_to_all_docs
def normalize_text(text: str) -> str:
    return text.lower()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def dedupe_by_embedding(docs: List[Document], embedding_model, threshold: float = 0.95) -> List[Document]:
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    kept_docs = []
    embeddings = []

    for doc in docs:
        if doc.metadata['content_len'] < 100:
            logging.info(f"{doc.metadata['source']} skipped for it is too short")
            continue
        text = doc.page_content.strip()
        emb = embedding_model.embed_documents([text])[0]
        if not embeddings:
            vector_store.add_documents([doc])
            embeddings.append(emb)
            kept_docs.append(doc)
            continue

        # делаем similarity search вручную: можно просто сравнить со всеми
        sims = [cosine_similarity(np.array(emb), np.array(e)) for e in embeddings]
        max_sim = max(sims)
        if max_sim < threshold:
            vector_store.add_documents([doc])
            embeddings.append(emb)
            kept_docs.append(doc)
        else:
            logging.info(f"Skipping duplicate (by embedding): '{text[:50]}...' с sim = {max_sim}")

    return kept_docs


def deduplicate_and_filter(threshold: float, docs: List[Document]) -> List[Document]:
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return dedupe_by_embedding(docs, embed_model, threshold=threshold)



def result_display(threshold: float):
    chain = (
        RunnableParallel(
            pdf=get_pdf_docs | clean_text | check_and_update_meta,
            md=get_md_docs | clean_text | check_and_update_meta
            )
        | RunnableLambda(lambda x: x["pdf"] + x["md"])
        | normalize_text
        | RunnableLambda(partial(deduplicate_and_filter, threshold))
    )
    result = chain.invoke(None)
    for doc in result:
        print(doc.page_content[:50])
        print('Источник:', doc.metadata['source'], doc.metadata['content_len'], '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and deduplicate documents.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for deduplication (lower = more aggressive dedup). Default: 0.95"
    )
    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0")
    result_display(threshold=args.threshold)