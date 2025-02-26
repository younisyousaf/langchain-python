from dotenv import load_dotenv
load_dotenv(".env")
import os 
os.environ['USER_AGENT'] = 'myagent'

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# docA = Document(
#     page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."
# )

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs =splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore= FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.4,
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's questions:
    Context: {context}
    Question: {input}
    """)

# chain = prompt | model

    chain = create_stuff_documents_chain(
    llm = model,
    prompt = prompt
    )
    
    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain 
    )
    
    return retrieval_chain

docs = get_documents_from_web("https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

# Select the first few chunks
docs = docs[:10]  

response = chain.invoke({
    "input": "Define LCEL?",
    "context": docs,
})

print(response["answer"])

