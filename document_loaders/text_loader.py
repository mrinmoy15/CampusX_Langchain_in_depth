from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.7)

prompt = PromptTemplate(
    template='Summarize the following poem: \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt|model|parser

loader = TextLoader('document_loaders/cricket.txt', encoding='utf-8')

docs = loader.load()

print(chain.invoke({'poem': docs[0].page_content}))


