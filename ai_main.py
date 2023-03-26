import subprocess
import pandas as pd
import numpy as np
import json
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import config
import openai
import pinecone
import datetime
import json


# Set the OpenAI API key
openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\""}]
selected_categories = []
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_chat_log(chat_log, filename):
    folder_path = "log/"
    with open(folder_path + filename, 'w') as f:
        json.dump(chat_log, f)

def clear_chat_history():
    global messages, session_id
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = [{"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\""}]


# Define the function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def load_index():
    pinecone.init(
       api_key =config.PINECONE_API_KEY,
        environment = config.PINECONE_environment )
    index_name = 'bricks'
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    return pinecone.Index(index_name)


def create_context(question,index, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the Pinecone index
    """

    # Get the embeddings for the question
    q_embed = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Query Pinecone index
    res = index.query(q_embed, filter={"category": {"$in": selected_categories}}, top_k=5, include_metadata=True)
           
    context_details = []
    returns = []
    cur_len = 0

    # Iterate through results, sorted by score (ascending), and add the text to the context until the context is too long
    for match in sorted(res['matches'], key=lambda x: x['score']):
        # Get the length of the text
        text_len = match['metadata']['n_tokens']
        
        # Add the length of the text to the current length
        cur_len += text_len + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else, add it to the text that is being returned
        returns.append(match['metadata']['content'])
        
        # Add the context information to the context_details list
        context_details.append({
            'score': match['score'],
            'category': match['metadata']['category'],
            'topic': match['metadata']['topic'],
            'url': match['metadata']['url'],
            'token': match['metadata']['n_tokens'],
        })
    print(f"How much Contexts found: {len(returns)} \n----------------\n ")
    # Return the context and context_details
    return "\n\n###\n\n".join(returns), context_details


def engineer_prompt(question, index):
    """
    Answer a question based on the most similar context from the Pinecone index
    """
    context, context_details = create_context(
        question,
        index)
    prompt =  [ {"role": "assistant", "content": f"Context: {context}"},
                {"role": "user", "content": f"{question}"}]
    return prompt, context_details


def answer_question(
    model="gpt-3.5-turbo",
    instruction = "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"",
    index = "",
    categories = [],
    question = "what?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop=None,
    callback=None):
   

    global messages, selected_categories
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    
    selected_categories = categories
    messages[0] = {"role": "system", "content": f"{instruction}"}
    prompt, context_details = engineer_prompt(question= question,index = index)
    messages += prompt
    print(f"messages before the prompting OpenAO: {messages} \n----------------\n ")
    # If debug, print the raw model response
    if debug:
        print(f"question = {question}")
        # print("Context:\n" + context)
        print(messages)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )
        output_text = ""
        for chunk in response:
            if "role" in chunk["choices"][0]["delta"]:
                continue
            elif "content" in chunk["choices"][0]["delta"]:
                r_text = chunk["choices"][0]["delta"]["content"]
                output_text += r_text
                if callback:
                    callback(r_text)
        if callback:
            system_message = {"role" : "assistant" , "content" :  output_text}
            messages = [d for d in messages if not (d.get("role") == "assistant" and "Context" in d.get("content"))]
            messages.append(system_message)
            print(f"messages after the prompt : {messages}  \n----------------\n")
            chat_log_filename = f"{session_id}_chat_log.json"
            save_chat_log(messages, chat_log_filename)
            print(f"{total_tokens =} {prompt_tokens =} { completion_tokens =}")
            return messages, context_details
        
        return response["choices"][0]["message"]['content'], context_details
    except Exception as e:
        print(e)
        print(f"something is wrong")
        return "", ""

