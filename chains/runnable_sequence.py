from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import \
(
    RunnableSequence, 
    RunnableParallel, 
    RunnablePassthrough, 
    RunnableBranch, 
    RunnableLambda
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0.7)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({'topic':'AI'}))

chain.get_graph().print_ascii()