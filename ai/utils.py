import json
import pinecone
import pandas as pd
import streamlit as st
from datetime import datetime
from google.oauth2 import service_account
from google.cloud import storage
from typing import List, Dict, Union, Tuple, Optional
import openai
import logging
import numpy as np
import os
from tqdm import tqdm
import tiktoken


session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)


# Define the function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def load_index(index_name: str, pinecone_api_key, pinecone_environment):
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    return pinecone.Index(index_name)


def load_from_gcs(bucket_name: str, file_path: str) -> Dict[str, any]:
    """
    Load a JSON file from Google Cloud Storage and convert it to a dictionary.

    Args:   bucket_name (str): The name of the GCS bucket.
            file_path (str): The file path within the bucket.

    Returns:  Dict[str, any]: The JSON content as a dictionary.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    content = blob.download_as_text()
    return json.loads(content)


def save_to_gcs(bucket_name: str, file_path: str, json_content: Dict[str, any]) -> None:
    """
    Save a dictionary as a JSON file to Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_path (str): The file path within the bucket.
        content (Dict[str, any]): The content to save as JSON.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_string(json_content, content_type="application/json")


def count_tokens(result):
    # Load the cl100k_base tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Convert the JSON response to a string
    result_str = json.dumps(result)

    # Tokenize the string and get the number of tokens
    n_tokens = len(tokenizer.encode(result_str))

    return n_tokens


def setup_logging(log_dir, log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_start_index(df, target_col):
    if df[target_col].notna().any():
        return df[target_col].isna().idxmax()
    return 0


def get_embedding(text, engine, sleep_time, logger):
    while True:
        try:
            return openai.Embedding.create(input=text, engine=engine)["data"][0]["embedding"]
        except openai.OpenAIError as e:
            logger.info(f"OpenAIError encountered. Waiting for 60 seconds before retrying.{e}")
            time.sleep(sleep_time)


def embed_dataframe(
    df,
    log_dir,
    checkpoint_file,
    checkpoint_interval,
    sleep_time,
    engine,
    text_col,
    target_col,
):
    if target_col not in df.columns:
        df[target_col] = np.nan
    df[target_col] = df[target_col].astype(object)

    logger = setup_logging(log_dir)

    start_index = get_start_index(df, target_col)
    print(f"{start_index =}")
    start_index = 0 if start_index is None else start_index

    for index in tqdm(range(start_index, df.shape[0]), total=df.shape[0], initial=start_index):
        try:
            df.at[index, target_col] = get_embedding(df.at[index, text_col], engine, sleep_time, logger)
            progress = f"Progress: {index + 1}/{df.shape[0]}"
            logger.info(progress)

            if (index + 1) % checkpoint_interval == 0:
                df.to_pickle(checkpoint_file)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"RateLimitError encountered for row {index}. Waiting for 60 seconds before retrying.")
                time.sleep(sleep_time)
                df.at[index, target_col] = get_embedding(df.at[index, text_col], engine, sleep_time, logger)
            else:
                logger.error(f"HTTPError {e.response.status_code} encountered for row {index}. Skipping.")

    df.to_pickle(checkpoint_file)

    return df


def serialize_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serializes all columns of a DataFrame using json.dumps.

    Args:
        df (pd.DataFrame): The DataFrame containing columns to serialize.

    Returns:
        pd.DataFrame: DataFrame with all columns serialized.
    """
    for column in df.columns:
        df[column] = df[column].apply(json.dumps)
    return df
