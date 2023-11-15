from .utils import embed_dataframe, count_tokens
from tqdm.notebook import tqdm
import copy
import tiktoken
import json
import pandas as pd
import re


def simplify_fitting_box_content(data):
    """Recursively simplifies the 'fitting_box_content' structure within the provided dictionary."""
    # Base cases
    if not isinstance(data, (dict, list)):
        return data

    if isinstance(data, list):
        new_data = []
        for item in data:
            new_data.append(simplify_fitting_box_content(item))
        return new_data

    if isinstance(data, dict):
        if "fitting_box_content" in data:
            # Transform the structure
            simplified_content = [entry[1] for entry in data["fitting_box_content"]]
            data["fitting_box_content"] = simplified_content
        for key, value in data.items():
            data[key] = simplify_fitting_box_content(value)
        return data


def chunking(json_data):
    chunks = {}

    json_data = simplify_fitting_box_content(json_data)
    print(f"within chunking {type(json_data) = }")
    # 1. Extracting vehicle section as a single chunk
    if "vehicle" in json_data:
        chunks["Fahrzeug-Informationen"] = json_data["vehicle"]

    # 2. Chunking the wheel section
    wheel_data = json_data.get("wheel", {})

    # Grouping 'rim_details' and 'description' together
    if "rim_details" in wheel_data or "description" in wheel_data:
        chunks["Felgen-Details und Beschreibung"] = {
            "rim_details": wheel_data.get("rim_details", {}),
            "description": wheel_data.get("description", ""),
        }

    # Creating separate chunks for other sections
    section_mappings = {
        "fitting_box_content": "Gutachtenerklaerung zur Passgenauigkeit",
        "expertise_requirement_information": "Anforderungen an das Felgen-Gutachten",
        "alternative_inches": "Alternative Felgengrößen",
        "alternative_colors": "Alternative Felgenfarben",
    }
    for key, new_key in section_mappings.items():
        if key in wheel_data:
            chunks[new_key] = wheel_data[key]

    # For 'expertise_information', we'll only flatten one level deep
    if "expertise_information" in wheel_data:
        flattened_data = {}
        nested_data = wheel_data["expertise_information"]
        for subkey, value in nested_data.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    flattened_data[f"{subkey}_{inner_key}"] = inner_value
            else:
                flattened_data[subkey] = value
        chunks["Informationen zum Felgen-Gutachten"] = flattened_data

    # For 'tyre_dimension_information', create a separate chunk for each tyre dimension
    if "tyre_dimension_information" in wheel_data:
        tyre_info = wheel_data["tyre_dimension_information"]
        for tyre_dimension, details in tyre_info.items():
            chunks[f"Informationen zur Reifendimension {tyre_dimension}"] = details

    return chunks


def embed_context_entry(temp_df_rim_details):
    # Initialize the main dataframe to hold all contexts and tokens
    df_dynamic_contexts = pd.DataFrame(columns=["rim_details_index", "context", "n_tokens"])
    matching_rows = temp_df_rim_details[(temp_df_rim_details["mapping_information"] != "None")]
    for index_rim_details, matching_row in matching_rows.iterrows():
        json_data = matching_row["mapping_information"]
        print(f"from within embed_context_entry: {type(json_data) =}")
        if json_data is None:
            continue
        # # result_flatten = flatten_json(json_data)
        # if index_rim_details == 5:
        #     save_json_to_file("test_json_orig", json_data)

        chunked_data = chunking(json_data)
        contexts = []
        n_tokens = []
        rim_details_indices = []
        # for key, value_dict in result_flatten.items():
        for key, value_dict in chunked_data.items():
            context = f"{key} : {value_dict}"
            n_tokens.append(count_tokens(context))
            contexts.append(context)
            rim_details_indices.append(index_rim_details)  # Append the current index for each context

        # Create a temporary DataFrame for the current iteration
        df_temp_context = pd.DataFrame(
            {
                "rim_details_index": index_rim_details,
                "context": contexts,
                "n_tokens": n_tokens,
            }
        )

        # Concatenate the temporary DataFrame with the main one
        df_dynamic_contexts = pd.concat([df_dynamic_contexts, df_temp_context], ignore_index=True)

        df_dynamic_contexts = embed_dataframe(
            df=df_dynamic_contexts,
            log_dir="log/template_embedding",
            checkpoint_file="data/checkpoint.pkl",
            checkpoint_interval=10,
            sleep_time=60,
            engine="text-embedding-ada-002",
            text_col="context",
            target_col="emb context",
        )
    return df_dynamic_contexts
