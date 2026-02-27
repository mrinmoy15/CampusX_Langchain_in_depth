from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You: \n")
    chat_history.append(HumanMessage(content=user_input)) # type: ignore
    if user_input.lower() in ['quit', 'exit']:
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) # type: ignore
    print(f"AI: \n {result.content}")
    
    
    