from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


def huggingface_chat_model_demo():

    llm = HuggingFacePipeline.from_model_id(
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task = "text-generation",
        model_kwargs = dict(
            temperature = 0.3,
            max_length = 64
        )
    )

    model = ChatHuggingFace(llm = llm)
    response = model.invoke("What is the capital of France?")
    print(response.content)
    

if __name__ == "__main__":
    huggingface_chat_model_demo()