# static web page
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question \n {question} \n from the following text - \n {text}',
    input_variables=['question','text']
)

url = 'https://docs.langchain.com/'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the url talks about?', 'text':docs[0].page_content}))