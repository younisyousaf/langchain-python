from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv(".env")

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
    # max_tokens=1000,
    # verbose=True
)

# response = llm.stream("Tell me a joke about chicken")
# for chunk in response:
#     print(chunk.content, end="", flush=True)

#Prompt Template
# prompt = ChatPromptTemplate.from_template("Tell me a joke about {subject}")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the list as a comma separated string."),
        ("human", "{subject}"),
    ]
)

# create LLM Chain
chain = prompt | llm

# response = chain.stream({"subject": "grudges"})
# for chunk in response:
#     print(chunk.content, end="", flush=True)

response = chain.invoke({"subject": "grudges"})
print (type(response.content))