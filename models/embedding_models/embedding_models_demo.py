from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def openai_embedding_model_demo():
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)

    result = embedding.embed_query("What is the capital of France?")
    print(len(result))
    print("===="*10)
    print(str(result))

def openai_batch_embedding_model_demo():
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)

    result = embedding.embed_documents(["What is the capital of France?", "What is the capital of Germany?"])
    print(len(result))
    print("===="*10)
    print(str(result))

def opensource_embedding_model_demo():
    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    result = embedding.embed_query("What is the capital of France?")
    print(len(result))
    print("===="*10)
    print(str(result))

def opensource_batch_embedding_model_demo():
    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    result = embedding.embed_documents(["What is the capital of France?", "What is the capital of Germany?"])
    print(len(result))
    print("===="*10)
    print(str(result))
    


if __name__ == "__main__":
    # openai_embedding_model_demo()

    # openai_batch_embedding_model_demo()

    # opensource_embedding_model_demo()

    opensource_batch_embedding_model_demo()