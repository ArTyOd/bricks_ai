from typing import Dict
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage
from typing import List, Dict, Union, Tuple, Optional
import streamlit as st
import json

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

import json  # Don't forget to import the json module


def initialize_dataframe(bucket_name: str, file_path: str, columns: List[str], dtypes: Dict[str, str] = None) -> pd.DataFrame:
    """
    Initialize a DataFrame by either loading it from Google Cloud Storage or creating a new one.

    Args:
        bucket_name: The name of the Google Cloud Storage bucket.
        file_path: The file path where the DataFrame is stored or will be stored in Google Cloud Storage.
        columns: The column names for initializing a new DataFrame.

    Returns:
        pd.DataFrame: The loaded or initialized DataFrame.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if blob.exists():
        content = blob.download_as_text()
        df = pd.read_json(content, orient="split")
        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)
    else:
        df = pd.DataFrame(columns=columns)
        if dtypes:
            for col, dtype in dtypes.items():
                df[col] = df[col].astype(dtype)
        json_content = df.to_json(orient="split")
        blob.upload_from_string(json_content, content_type="application/json")

    return df


@st.cache_data()
def initialize_dataframe_cache(bucket_name: str, file_path: str, columns: List[str], dtypes: Dict[str, str] = None) -> pd.DataFrame:
    """
    This function is specifically for caching df_all_rims DataFrame.
    """
    return initialize_dataframe(bucket_name, file_path, columns, dtypes)


def initialize_all_dataframes(bucket_name: str, folder_path: str, config_file_name: str) -> Dict[str, pd.DataFrame]:
    """
    Initialize multiple DataFrames by either loading them from Google Cloud Storage or creating new ones.

    Args:
        bucket_name: The name of the Google Cloud Storage bucket.
        folder_path: The folder path within the bucket.
        config_file_name: The name of the JSON file containing DataFrame configurations.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of initialized DataFrames.
    """
    # Load DataFrame configuration from the provided config file name
    config_file_path = f"{folder_path}/{config_file_name}"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(config_file_path)
    if blob.exists():
        content = blob.download_as_text()
        dataframes = json.loads(content)
    else:
        raise FileNotFoundError(f"Configuration file {config_file_path} not found in bucket {bucket_name}.")

    initialized_dfs = {}
    for df_name, config in dataframes.items():
        columns = config.get("columns", [])
        dtypes = config.get("dtypes", {})
        file_path = f"{folder_path}/{df_name}.json"

        if df_name == "df_all_rims":
            continue
            # initialized_dfs[df_name] = initialize_dataframe_cache(bucket_name, file_path, columns, dtypes)
        else:
            initialized_dfs[df_name] = initialize_dataframe(bucket_name, file_path, columns, dtypes)

    return initialized_dfs
