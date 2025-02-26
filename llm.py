from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(".env")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    # max_tokens=1000,
    # verbose=True
);

# response = llm.invoke("Hello, how are you?")
# response = llm.batch(["Hi","Write a POEM about AI"])
response = llm.stream("Write a POEM about AI")
for chunk in response:
    print(chunk.content, end="", flush=True)
# print(response)