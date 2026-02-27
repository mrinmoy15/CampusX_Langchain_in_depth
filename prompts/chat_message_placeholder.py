from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt_template = ChatPromptTemplate(
    [
        ('system', 'You are a helpful customer support rep'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

chat_history = [
    HumanMessage(content="Hi, I placed an order last week and I'd like to request a refund. My order number is #45823."),
    AIMessage(content="Hello! I'm sorry to hear you'd like a refund. I'd be happy to help you with that. Could you please tell me the reason for your refund request?"),
    HumanMessage(content="I received the wrong item. I ordered a blue jacket in size L but got a red one in size M."),
    AIMessage(content="I sincerely apologize for that mistake. I've pulled up your order #45823 and confirmed the error. I'll initiate a full refund to your original payment method within 5–7 business days. You'll also receive a prepaid return label via email shortly."),
]

# create prompt
prompt = prompt_template.invoke(
    {
        'chat_history': chat_history,
        'query': "Okay, I want to get my refund as soon as possible."
    }
)

print(llm.invoke(prompt).content)