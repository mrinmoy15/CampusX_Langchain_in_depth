from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(gt = 18, description="person's age")
    city: str = Field(description="person's city")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | chat_model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)