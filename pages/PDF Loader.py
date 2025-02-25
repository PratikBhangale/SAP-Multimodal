import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from dotenv import load_dotenv


import uuid
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")


from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Configure your embedding model and vector store
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embedding,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
print("Astra vector store configured")

# Set the title of the Streamlit app
st.set_page_config(layout="wide", page_title="Creating a Data Store")
st.title("Creating a Data Store")

# Define the function to handle document embeddings
def vector_embedding(uploaded_files):

    st.session_state.embeddings = OpenAIEmbeddings()

    # docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to disk
        file_path = os.path.join("uploaded_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


        all_chunks = []  

        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables

            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,

            # extract_images_in_pdf=True,          # deprecated
        )
        all_chunks.extend(chunks)

    # separate tables from texts
    tables = []
    texts = []

    for chunk in all_chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    # Get the images from the CompositeElement objects
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64

    images = get_images_base64(all_chunks)

    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    # model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    model = Ollama(model='llama3.1', temperature=0.2)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})


    # Summarise Images
    prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    image_summaries = chain.batch(images)





    id_key = "undefined"

    # Add texts
    # doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={"doctype":"text"}) for i, summary in enumerate(text_summaries)
    ]
    if summary_texts:
        vstore.add_documents(summary_texts)


    # Add tables
    # table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={"doctype":"table"}) for i, summary in enumerate(table_summaries)
    ]
    if summary_tables:
        vstore.add_documents(summary_tables)

    # Add image summaries
    # img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={"doctype":"image"}) for i, summary in enumerate(image_summaries)
    ]
    if summary_img:
        vstore.add_documents(summary_img)



# Allow users to upload PDF documents
uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])

# Button to create document embeddings
if st.button("Create Document Embeddings") and uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    vector_embedding(uploaded_files)
    st.write("Vector Store DB is ready")

