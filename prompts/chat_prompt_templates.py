from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms what is the {topic}.")
])

prompt = chat_template.invoke({"domain": "cricket", "topic":"Dusra"})

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

print(chat_model.invoke(prompt).content)

