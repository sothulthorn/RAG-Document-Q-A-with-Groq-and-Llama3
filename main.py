## RAG Document Q&A Application
## Uses Groq (Llama3) as the LLM and HuggingFace embeddings with FAISS
## for retrieval-augmented generation over PDF research papers.

import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

## Load the GROQ API key from environment
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

## Initialize the Groq LLM with Llama3-8b model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

## Define the prompt template that instructs the LLM to answer
## only from the provided context (retrieved documents)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    """Load PDFs, split into chunks, generate embeddings, and store in FAISS vector DB.
    Uses Streamlit session_state to persist across reruns and avoid reprocessing."""
    if "vectors" not in st.session_state:
        ## Initialize HuggingFace embedding model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ## Load all PDFs from the research_papers directory
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        ## Split documents into chunks of 1000 chars with 200 char overlap for better retrieval
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        ## Create FAISS vector store from the document chunks
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

## Streamlit UI
st.title("RAG Document Q&A With Groq And Lama3")

user_prompt = st.text_input("Enter your query from the research paper")

## Button to trigger the embedding/indexing pipeline
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

## When a user query is submitted, build the retrieval chain and get the answer
if user_prompt:
    ## Create a chain that stuffs retrieved documents into the prompt context
    document_chain = create_stuff_documents_chain(llm,prompt)
    ## Get a retriever from the FAISS vector store
    retriever = st.session_state.vectors.as_retriever()
    ## Combine the retriever and document chain into a full RAG pipeline
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    ## Invoke the chain and measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    ## Display the LLM's answer
    st.write(response['answer'])

    ## Show the retrieved source documents in a collapsible section
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
