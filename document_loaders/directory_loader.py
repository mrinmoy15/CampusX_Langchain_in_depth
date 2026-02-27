from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='document_loaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader # type: ignore
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)