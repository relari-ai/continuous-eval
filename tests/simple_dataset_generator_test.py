import datetime
import os

import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone

from continuous_eval.simple_dataset_generator import SimpleDatasetGenerator

load_dotenv()


def load_data_from_pinecone():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    index_name = "pgessays"
    pinecone_index = Pinecone.from_existing_index(index_name=index_name, embedding=OpenAIEmbeddings())

    return pinecone_index


def load_data_into_chromadb(directory_path):
    # Customize to load various types of documents. Default loads txt.
    loader = DirectoryLoader(directory_path, glob="**/*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    print(f"Finished loading {len(docs)} chunks from {len(documents)} documents into ChromaDB. Generating dataset...")
    return db


def generate_data_from_documents(
    documents, generator_llm="gpt-4-1106-preview", num_questions=10, multi_hop_precentage=0.2, max_try_ratio=3
):
    db = load_data_into_chromadb(documents)
    return generate_data_from_vectordb(db, 1536, generator_llm, num_questions, multi_hop_precentage, max_try_ratio)


def generate_data_from_vectordb(
    vector_db_index,
    index_dimension,
    generator_llm="gpt-4-1106-preview",
    num_questions=10,
    multi_hop_precentage=0.2,
    max_try_ratio=3,
):
    dataset_generator = SimpleDatasetGenerator(
        vector_store_index=vector_db_index,
        generator_llm=generator_llm,
    )

    results = dataset_generator.generate(
        index_dimension=index_dimension,
        num_questions=num_questions,
        multi_hop_percentage=multi_hop_precentage,
        max_try_ratio=max_try_ratio,
    )
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "tests/data/generated_dataset"
    os.makedirs(output_directory, exist_ok=True)
    output_csv = (
        f"{output_directory}/G_{generator_llm}_Q_{num_questions}_MH%_{multi_hop_precentage}_{current_datetime}.csv"
    )
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return output_csv


file_path = "tests/data/generated_dataset/graham_essays"
# db = load_data_into_chromadb(file_path)
# output_csv = generate_data_from_vectordb(db)
output_csv = generate_data_from_documents(file_path)
print(f"Generated: {output_csv}")
