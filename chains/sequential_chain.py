from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

parser = StrOutputParser()

chain = prompt1|chat_model|parser|prompt2|chat_model|parser

result = chain.invoke({"topic": "Jan6 Insurrection"})

print(result)

chain.get_graph().print_ascii()
