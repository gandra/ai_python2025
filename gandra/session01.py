# # GenAI/RAG in Python 2025
#
# ## Session 01. A Basic RAG Framework

import os
import numpy as np
import pandas as pd
from openai import OpenAI


def load_data_frame(file_path: str):
    # ## 1. Let's grab some text... Italian cuisine, for example?

    # Load the CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Display some basic information
    print(df.info())
    print(df.head())

    return df

def get_clilent():
    api_key = os.getenv("OPENAI_API_KEY")

    # Instantiate the OpenAI client with your API key
    client = OpenAI(api_key=api_key)

    return client


def get_embeddings_data_frame(file_path: str):
    # Select the embedding model to use (as per OpenAI docs)
    model_name = "text-embedding-3-small"

    # Prepare a list to collect embedding vectors
    embeddings = []

    # Iterate over each row in your DataFrame `df`
    client = get_clilent()
    df = load_data_frame(file_path)
    for idx, row in df.iterrows():
        # grab the receipt text for this row
        text = row["receipt"]
        # If it's not a valid string, skip embedding
        if not isinstance(text, str) or text.strip() == "":
            embeddings.append(None)
            continue

        # Call the embeddings endpoint on the client
        resp = client.embeddings.create(
            model=model_name,
            input=[text]
        )

        # Extract the embedding vector from the response object
        emb = resp.data[0].embedding

        # Append that embedding vector to our list
        embeddings.append(emb)

    # After the loop, assign embeddings list to a new DataFrame column
    df["embedding"] = embeddings

    # Show first few rows to verify
    df.head()

    print(type(df['embedding'][0]))

    print(len(df['embedding'][0]))

    return df



def get_recipes_for_prompt(user_text: str):
    # Path to the CSV file
    file_path = "../_data/italian_recipes_clean.csv"

    # ## 2. We need to embed the text in the same way as the examples
    model_name = "text-embedding-3-small"      # Select the embedding model to use (as per OpenAI docs)
    client = get_clilent()
    # ... and of course we need an embedding of `user_text` as well:
    resp = client.embeddings.create(
            model=model_name,
            input=[user_text]
        )
    user_query = resp.data[0].embedding

    print(type(user_query))
    print(len(user_query))

    # ## 4. Find the most suitable examples that match the user input

    # scipy has a function to compute cosine distance: cosine()
    from scipy.spatial.distance import cosine

    # Compute similarity scores: similarity = 1 − cosine_distance
    scores = []
    df = get_embeddings_data_frame(file_path)
    for emb in df["embedding"]:
        if emb is None:
            scores.append(-1.0)
        else:
            scores.append(1.0 - cosine(np.array(emb), np.array(user_query)))

    # Get top 5 indices
    top5 = np.argsort(scores)[-5:]
    # N.B. np.argsort(scores) — returns an array of indices that would
    # sort scores in ascending order.
    # [-5:] — takes the last 5 indices from that sorted‐indices array.
    # Since the full array is in ascending order, its last 5 indices correspond to
    # the 5 highest scores.

    # Build a single output string with titles and recipes
    output_lines = []
    for i in top5:
        title = df.iloc[i]["title"]
        recipe = df.iloc[i]["receipt"]
        output_lines.append(f"{title}:\n{recipe}")
    prompt_recipes = "\n\n".join(output_lines)

    print(prompt_recipes)

    # $$
    # \cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\;\|\mathbf{b}\|}
    # = \frac{\sum_{i=1}^n a_i\,b_i}{\sqrt{\sum_{i=1}^n a_i^2}\;\sqrt{\sum_{i=1}^n b_i^2}}
    # $$
    #
    # A common definition of **cosine similarity** is:
    #
    # $$
    # d_{\text{cos}}(\mathbf{a},\mathbf{b}) = 1 - \cos\theta
    # $$
    #
    # - In text / embedding applications, higher cosine similarity (or lower cosine distance) means vectors are more semantically aligned.
    #

    # ## 5. Finally, use an LLM to shape the final response

    prompt = f"""
    You are a helpful Italian cooking assistant.  
    Here are some recipe examples I found that may or may not be relevant to the user's request:
    
    {prompt_recipes}

    User’s question: "{user_text}"
    
    From the examples above:
    1. Determine which recipes are *relevant* to what the user asked and which are not.
    2. Discard or ignore irrelevant ones, and focus on relevant ones.
    3. For each relevant example, rephrase the recipe in a more narrative, 
    conversational style, adding cooking tips, alternative ingredients, variations, 
    or suggestions.
    4. Then produce a final response to the user: a narrative that weaves 
    together those enhanced recipes (titles + steps + tips) in an engaging way.
    5. Don't forget to use the original titles of the recipes.
    6. Advise on more than one recipe - if there are more than one relevant!
    
    Do not just list recipes — tell a story, connect to the user's question, 
    and use the examples as inspirations, but enhance them.  
    Make sure your response is clear, helpful, and focused on what the user wants.
    """

    response = client.chat.completions.create(
        model="gpt-4",    # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are a helpful Italian cooking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=5000
    )

    reply_text = response.choices[0].message.content
    print(reply_text)




# ## 3. Now we need a user input...
user_text = """
Hi! I’d like to cook a good Italian dish for lunch! I have potatoes, carrots, 
rosemary, and pork. Can you recommend a recipe and help me a bit with 
preparation tips?
"""

get_recipes_for_prompt(user_text)