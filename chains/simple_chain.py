from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

parser = StrOutputParser()

chain = template|chat_model|parser

result = chain.invoke({"topic": "Femdom"})

print(result)