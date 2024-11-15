import os
import pandas as pd
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env")

# Environment variables
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

# Set up logger
logger = get_logger(__name__)

# Load embedding model
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# Initialize LLM
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# StreamHandler to display response
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Main application
def main():
    st.header("📄Chat with your CSV file")

    # Upload your CSV file
    csv_file = st.file_uploader("Upload your CSV", type="csv")

    if csv_file is not None:
        # Read CSV content
        df = pd.read_csv(csv_file)

        # Convert the DataFrame to a text string (this can be modified based on which data you want to use)
        text = df.to_string()

        # langchain text splitter to split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="csv_bot",  # Updated to reflect CSV data
            node_label="CsvBotChunk",  # Updated node label
            pre_delete_collection=True,  # Delete existing CSV data
        )
        
        # Set up RetrievalQA with the vectorstore
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user queries/questions
        query = st.text_input("Ask questions about your CSV file")

        if query:
            # Handle response stream
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()
