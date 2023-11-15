from .context_retrieval import context_retrieval_entry
from .embed_context import embed_context_entry
from .document_retrieval import gather_contexts, create_context_pinecone
from .llm_interactions import answer_question, answer_question_streaming
from .initialize import initialize_all_dataframes
from .utils import load_from_gcs, save_to_gcs, load_index

import json
import pandas as pd
import pinecone
import streamlit as st
from typing import List, Dict, Tuple, Optional, Union, Callable


def initialize_process_request(bucket_name, folder_path, config_file_name):
    # Initialization logic here
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_environment = st.secrets["PINECONE_environment"]
    pinecone_index = load_index(index_name="fo", pinecone_api_key=pinecone_api_key, pinecone_environment=pinecone_environment)

    return main(bucket_name, folder_path, config_file_name, pinecone_index)


def main(bucket_name, folder_path, config_file_name, pinecone_index):
    dfs = initialize_all_dataframes(bucket_name, folder_path, config_file_name)
    df_file_paths = {df_name: f"{folder_path}/{df_name}.json" for df_name in dfs.keys()}

    df_requests = dfs["df_requests"]
    df_rim_details = dfs["df_rim_details"]
    # df_dynamic_contexts = dfs["df_dynamic_contexts"]

    def process_request(
        # function parameters
        text="",
        instruction="",
        categories="",
        debug=False,
        model="gpt-4",
        temperature=0.3,
        index=pinecone_index,
    ):
        nonlocal df_requests, df_rim_details
        print(f"{text = }\n")
        if debug:
            print(f"Debug mode on.\n")
            temp_df_requests = pd.read_pickle("data/temp_df_requests.pkl")
            temp_df_rim_details = pd.read_pickle("data/temp_df_rim_details.pkl")
        else:
            temp_df_requests, temp_df_rim_details = context_retrieval_entry(
                text=text,
                instruction=instruction,
                debug=False,
                df_requests=df_requests,
                df_rim_details=df_rim_details,
            )

            # temp_df_requests.to_pickle("data/temp_df_requests.pkl")
            # temp_df_rim_details.to_pickle("data/temp_df_rim_details.pkl")

        df_requests = pd.concat([df_requests, temp_df_requests], ignore_index=True)
        df_rim_details = pd.concat([df_rim_details, temp_df_rim_details], ignore_index=True)

        save_to_gcs(
            bucket_name,
            df_file_paths["df_requests"],
            df_requests.to_json(orient="split"),
        )

        save_to_gcs(
            bucket_name,
            df_file_paths["df_rim_details"],
            df_rim_details.to_json(orient="split"),
        )

        df_dynamic_contexts = embed_context_entry(temp_df_rim_details)

        row = temp_df_requests.iloc[0]
        question = row["question"]
        matching_rows = temp_df_rim_details[(temp_df_rim_details["mapping_information"].notnull())]

        print(f"{matching_rows['mapping_information'] = }\n")

        specific_context = gather_contexts(question, matching_rows, df_dynamic_contexts, token_limit=4000)
        print(f"{specific_context = }\n")

        general_context, general_context_details = create_context_pinecone(question, pinecone_index, max_len=1500, size="ada")
        print(f"{general_context = }\n")
        prompt = [
            {
                "role": "assistant",
                "content": f"[Vehicle Wheel Specific Context]: {specific_context}",
            },
            {"role": "assistant", "content": f"[General Context]: {general_context}"},
            {"role": "user", "content": f"[Question]: {question}"},
        ]
        # print(f"{prompt = }")
        answer_instruction = instruction["gpt-response1"]
        print(f"{answer_instruction = }")
        # response, used_tokens = answer_question(
        #     model=model, instruction=answer_instruction, prompt=prompt, debug=False
        # )

        # temp_df_requests.at[0, "gpt4"] = response
        # temp_df_requests.at[0, "context"] = specific_context

        # temp_df_requests.at[0, "general_context"] = general_context
        # temp_df_requests.at[0, "general_context_details"] = general_context_details
        # temp_df_requests.to_pickle(checkpoint_file)
        # print(f"{question = }\n")
        # print(f"{response = }\n")
        # print(f"{used_tokens = } \n--------------------\n ")

        save_to_gcs(
            bucket_name,
            df_file_paths["df_dynamic_contexts"],
            df_dynamic_contexts.to_json(orient="split"),
        )

        return prompt, answer_instruction

    return process_request


if __name__ == "__main__":
    bucket_name = "bucket_g_cloud_service_1"
    folder_path = "fo"
    config_file_name = "dataframes.json"

    import openai

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = openai_api_key

    process_request = initialize_process_request(bucket_name, folder_path, config_file_name)

    # Test code
    with open("tests/fixtures/sample_question_1.txt", "r", encoding="utf-8") as file:
        user_input = file.read()

    with open("data/instructions.json", "r") as file:
        instruction = json.load(file)

    final = process_request(
        text=user_input,
        instruction=instruction,
        debug=False,
    )
    print(final)
