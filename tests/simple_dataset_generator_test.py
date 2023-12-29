import datetime
import os

import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from continuous_eval.simple_dataset_generator import SimpleDatasetGenerator

load_dotenv()


def generate_data():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    index_name = "pgessays"

    openaiembeddings = OpenAIEmbeddings()
    pinecone_pg_essays_index = Pinecone.from_existing_index(index_name=index_name, embedding=openaiembeddings)

    generator_llm = "gpt-4-1106-preview"
    num_questions = 1
    multi_hop_precentage = 0.2

    dataset_generator = SimpleDatasetGenerator(
        vector_store_index=pinecone_pg_essays_index,
        generator_llm=generator_llm,
    )

    results = dataset_generator.generate(
        index_dimension=1536, num_questions=num_questions, multi_hop_percentage=multi_hop_precentage
    )
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "tests/data/generated_dataset"
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    output_csv = (
        f"{output_directory}/G_{generator_llm}_Q_{num_questions}_MH%_{multi_hop_precentage}_{current_datetime}.csv"
    )
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return output_csv


output_csv = generate_data()
print(f"Generated: {output_csv}")
