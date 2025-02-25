import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langsmith import traceable


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# Remove the export statements and set environment variables using os.environ
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"  # Replace <your-api-key> with the actual key
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "SAP_RAG"



# Creating an Array for storing the chat history for the model.
context = []


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LANG-CHAIN CHAT WITH LLAMA")

# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Create the Sidebar
sidebar = st.sidebar

# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    context = []
    st.session_state.messages =[]



# Defining out LLM Model
# llm = Ollama(model='llama3.1', temperature=0)
llm = ChatOpenAI()


# Configure your embedding model and vector store
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embedding,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
retriever = vstore.as_retriever(search_kwargs={"k": 3})


# Create the function to stream output from llm
def get_response(question):

    # Define the prompt template
    prompt_template = """You are a helpful assistant. You should answer the following question or questions as best you can using the context provided.

    Context:{context}

    Question: {question}
    answer"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the chain
    # chain = PROMPT | llm | StrOutputParser()
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return a streaming output.
    # return chain.stream({
    #     "context": context,
    #     "question": question,
    # })
    return chain.invoke(question)

# ------------------------------------------------------------------------------------------------------------------------------
@traceable
def start_app():

        try:
            OLLAMA_MODELS = ollama.list()["models"]
        except Exception as e:
            st.warning("Please make sure Ollama is installed and running first. See https://ollama.ai for more details.")
            st.stop()

        question = st.chat_input("Ask Anything.", key=1)

        if question:


            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])


            st.session_state.messages.append({"role": "user", "content": question})

            # response = get_response(PROMPT, llm).invoke({"input": question, "context": context})
            with st.chat_message("Human"):
                st.markdown(question)

            with st.chat_message("AI"):
                with st.spinner("Thinking"):
                    response = st.write(get_response(question))


            st.session_state.messages.append({"role": "assistant", "content": response})


            for message in st.session_state.messages:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

            context.append(HumanMessage(content=question))
            context.append(AIMessage(content=str(response)))

            # for message in st.session_state.messages:
            #     with st.chat_message(message["role"]):
            #         st.write(message["content"])



if __name__ == "__main__":
    start_app()