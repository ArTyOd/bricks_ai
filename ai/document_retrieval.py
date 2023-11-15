import itertools
import openai

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def map_dynamic_context(
    question,
    df_or_path="data/fo/df_dynamic_context.pkl",
    engine_name="text-embedding-ada-002",
    emb_col_name="emb hersteller design",
    text_col_name="hersteller design",
    threshold=0.75,
):
    """
    Create a context for a question by finding the most similar context from the summaries DataFrame.
    The function outputs only the id, content column, and score.
    """

    #
    # Check if df_or_path is a DataFrame or a string (path) and load DataFrame
    if isinstance(df_or_path, pd.DataFrame):
        df_dynamic_contexts = df_or_path
    else:
        df_dynamic_contexts = pd.read_pickle(df_requests_or_path)

    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine=engine_name)["data"][0]["embedding"]

    # Prepare embeddings from df_dynamic_contexts for comparison
    embeddings = np.array(df_dynamic_contexts[emb_col_name].tolist())

    # Calculate cosine similarities
    similarities = cosine_similarity([q_embed], embeddings)[0]

    # Add similarities to the DataFrame
    df_dynamic_contexts["similarity"] = similarities

    # Sort DataFrame by similarity in descending order
    sorted_df = df_dynamic_contexts.sort_values(by="similarity", ascending=False)

    context_details = []
    returns = []
    cur_len = 0

    for index, row in sorted_df.iterrows():
        # Get the length of the text
        text_len = row["n_tokens"]

        # Add the length of the text to the current length
        cur_len += text_len + 4

        # If the context is too long or below threshold, break
        if row["similarity"] < threshold:
            break

        # Else, add it to the text that is being returned
        yield {
            "rim_details_index": row["rim_details_index"],
            "score": row["similarity"],
            "context": row["context"],
            "n_tokens": row["n_tokens"],
        }


def gather_contexts(question, matching_rows, df_dynamic_contexts, token_limit=4000):
    """
    Gather contexts from different generators alternatively while maintaining a token limit.

    Parameters:
    - question: The question or prompt for which context is sought.
    - matching_rows: DataFrame rows that match the current question/request.
    - df_dynamic_contexts: DataFrame containing dynamic contexts.
    - token_limit: Maximum token limit for the accumulated contexts.

    Returns:
    - final_context: Dictionary with the accumulated contexts for the provided question.
    """

    # Step 1: Create a list of context generators for each rim_details index
    generators = []
    keys = []

    for index_rim_details, matching_row in matching_rows.iterrows():
        print(f"{index_rim_details = }")

        filtered_df = df_dynamic_contexts[df_dynamic_contexts["rim_details_index"] == index_rim_details]
        print(f"{filtered_df.shape = }")
        context_generator = map_dynamic_context(
            question=question,
            df_or_path=filtered_df,
            emb_col_name="emb context",
            text_col_name="context",
            threshold=0.75,
        )
        generators.append(context_generator)

        key_name = f'{matching_row["hersteller"]} {matching_row["design"]}'
        keys.append(key_name)

    # Step 2: Alternate between generators to pull contexts
    accumulated_tokens = 0
    final_context = {key: [] for key in keys}
    gen_len = len(generators)
    i = 0  # Using explicit index to control the loop

    while accumulated_tokens < token_limit and gen_len > 0:
        generator = generators[i]
        key = keys[i]

        # Try to get the next context without consuming the generator
        next_ctx = list(itertools.islice(generator, 1))

        # If generator is exhausted, break out of the loop
        if not next_ctx:
            break

        ctx = next_ctx[0]
        accumulated_tokens += ctx["n_tokens"]

        if accumulated_tokens <= token_limit:  # Ensure we don't go over the limit
            final_context[key].append(ctx["context"])
        else:  # If we've gone over the limit, roll back the addition and break
            accumulated_tokens -= ctx["n_tokens"]
            break

        i = (i + 1) % gen_len  # Move to the next generator in a cyclic manner

    return final_context


def create_context_pinecone(question, selected_categories, index, max_len=1500, size="ada"):
    """
    Create a context for a question by finding the most similar context from the Pinecone index
    """
    print(f"{question =}")
    print(f"{index =}")
    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine="text-embedding-ada-002")["data"][0]["embedding"]

    # Query Pinecone index
    # res = index.query(q_embed, top_k=10, include_metadata=True)
    res = index.query(q_embed, filter={"category": {"$in": selected_categories}}, top_k=5, include_metadata=True)

    context_details = []
    returns = []
    cur_len = 0

    # Iterate through results, sorted by score (ascending), and add the text to the context until the context is too long
    for match in sorted(res["matches"], key=lambda x: x["score"]):
        # Get the length of the text
        text_len = match["metadata"]["n_tokens"]

        # Add the length of the text to the current length
        cur_len += text_len + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else, add it to the text that is being returned
        returns.append(match["metadata"]["content"])

        # Add the context information to the context_details list
        context_details.append(
            {
                "score": match.get("score", None),
                "category": match["metadata"].get("category", None),
                "topic": match["metadata"].get("topic", None),
                "question": match["metadata"].get("question", None),
                "content": match["metadata"].get("content", None),
                "url": match["metadata"].get("url", None),
                "token": match["metadata"].get("n_tokens", None),
            }
        )
    print(f"How much Contexts found: {len(returns)} \n----------------\n ")
    # Return the context and context_details
    return "\n\n###\n\n".join(returns), context_details
