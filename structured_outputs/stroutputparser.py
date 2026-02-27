from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]

)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text: \n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1|chat_model|parser|template2|chat_model|parser

result = chain.invoke({"topic": "BDSM"})

print(result)