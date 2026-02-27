from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = \
        Field(description="Return sentiment of the review either positive, negative, or neutral")



parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2|model|parser), # type: ignore
    (lambda x: x.sentiment == 'negative', prompt3|model|parser), # type: ignore
    RunnableLambda(lambda x: "Neutral feedback")
)

chain = classifier_chain | branch_chain
result =chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)

chain.get_graph().print_ascii()



