from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
model2 = ChatOpenAI(model="gpt-4o", temperature=0.7)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text: \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']

)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1|model1|parser,
        "quiz": prompt2|model2|parser

    }
)

merge_chain = prompt3|model1|parser

final_chain = parallel_chain|merge_chain

text = """
Virtualization vs Containerization
===================================
Both virtualization and containerization enable running multiple workloads on the same physical hardware, but they differ in architecture, performance, and use cases.

Virtualization uses a hypervisor to create Virtual Machines (VMs), each with its own guest operating system, virtualized hardware, and applications. This provides strong isolation and OS flexibility, allowing different OS types on the same host. However, VMs are resource-heavy, have slower startup times, and require more storage and memory.

Containerization, on the other hand, is OS-level virtualization. Containers share the host OS kernel but run applications in isolated user spaces with their dependencies and configurations. They are lightweight, start quickly, and are highly portable across environments, making them ideal for microservices and cloud-native deployments.

Key Differences:

Isolation: Virtualization → Full OS isolation per VM. Containerization → Process-level isolation, shared kernel.

Resource Usage: VMs → Higher overhead due to multiple OS instances. Containers → Lower overhead, efficient resource sharing.

Performance & Startup: VMs → Slower boot due to OS load. Containers → Near-instant startup.

Portability: VMs → Less portable, OS-dependent images. Containers → Highly portable across compatible OS hosts.

Use Cases: VMs → Legacy apps, multi-OS environments, strong security isolation. Containers → Microservices, CI/CD pipelines, scalable cloud apps.

"""

result = final_chain.invoke({"text": text})

print(result)

final_chain.get_graph().print_ascii()



