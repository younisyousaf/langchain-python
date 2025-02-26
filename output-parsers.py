from dotenv import load_dotenv
load_dotenv(".env")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7,
)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system","What is HIPAA?"),
        ("human","{subject}"),
    ])
    parser = StrOutputParser()
    chain = prompt | model | parser

    return chain.invoke({"subject": "grudges"})

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Generate a list of 10 synonyms for the following word. Return the list as a comma separated string."),
        ("human","{subject}"),
    ])
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser

    return chain.invoke(
        {"subject": "grudges"}
    )

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human","{phrase}"),
    ])
    
    class Person(BaseModel):
        recipe: str = Field(description="The name of the recipe")
        ingredients: list = Field(description="ingredients")
        
    parser = JsonOutputParser(pydantic_object=Person )
    chain = prompt | model | parser

    return chain.invoke({"phrase": "The ingredients for a Margherita Pizza are tomato sauce, mozzarella, onions, tomatoes, and basil", "format_instructions": parser.get_format_instructions()})
print(call_json_output_parser())
# print(call_list_output_parser())
# print(call_string_output_parser())