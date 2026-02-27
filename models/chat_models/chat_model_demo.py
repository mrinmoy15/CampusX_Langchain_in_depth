from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def openai_chat_model_demo():
    model = ChatOpenAI(model = 'gpt-4', temperature = 1.5, max_completion_tokens = 10)

    response = model.invoke("What is the capital of France?")

    print(response.content)


if __name__ == "__main__":
    openai_chat_model_demo()