from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

result = llm.invoke("What is the capotal of Franch?")

print(result)