import re
import bs4
from typing import List

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document


class LoaderRunnable(RunnableLambda):
    def __init__(self, loader):
        super().__init__(lambda _: list(loader.lazy_load()))


def apply_func_to_all_docs(func):
    def process_docs(docs):
        for doc in docs:
            doc.page_content = func(doc.page_content)
        print(docs)
        return docs
    return process_docs


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


def check_and_update_meta(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.metadata.update({"content_len": len(doc.page_content)})
    return docs


def normalize_text(text: str) -> str:
    return text.lower()


def deduplicate_and_filter(docs: List[Document]) -> List[Document]:
    return [doc for doc in docs if  len(doc.metadata['content_len']) < 400]


def result_display():
    load_pdf = LoaderRunnable(PyMuPDFLoader("/home/mike/PycharmProjects/index_and_embeddings/clean_and_normalization/docs/Deep_Purple_Mk_I_1968-1969.pdf"))
    load_html = LoaderRunnable(WebBaseLoader(
                    web_paths=("https://docs.langchain.com/oss/python/langchain/overview",),
                    bs_kwargs={"parse_only": bs4.SoupStrainer(id="content")}))
    chain = (
        RunnableParallel(
            pdf=load_pdf | RunnableLambda(apply_func_to_all_docs(clean_text)) | RunnableLambda(check_and_update_meta),
            html=load_html | RunnableLambda(apply_func_to_all_docs(clean_text)) | RunnableLambda(check_and_update_meta)
            )
                     | RunnableLambda(lambda x: x["pdf"] + x["html"])
                     | RunnableLambda(apply_func_to_all_docs(normalize_text))
                     | RunnableLambda(deduplicate_and_filter)
    )
    result = chain.invoke(None)
    for doc in result:
        print(doc.page_content[:50])
        print('Источник:', doc.metadata['source'], doc.metadata['content_len'], '\n')


if __name__ == "__main__":
    result_display()