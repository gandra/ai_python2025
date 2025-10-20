# Session 01 — Basic RAG Framework Explained

This notebook walks through how to build a very small **Retrieval-Augmented Generation (RAG)** pipeline in Python. A RAG system first *retrieves* helpful reference texts and then uses a **Large Language Model (LLM)** to *generate* a final answer that blends the retrieved knowledge with the user’s question. Everything in this walkthrough is written with a first-year informatics student in mind, so we’ll unpack each idea one by one.

---

## 1. Imports and setup

```python
import os
import numpy as np
import pandas as pd
from openai import OpenAI
```

- `os` gives us tools to work with the operating system, such as reading environment variables (useful for secrets like API keys).
- `numpy` (shortened to `np`) is the fundamental package for working with numbers and vectors in Python.
- `pandas` (shortened to `pd`) offers `DataFrame`, a table-like data structure perfect for working with CSV files.
- `OpenAI` is the Python SDK class used to send requests to OpenAI’s APIs (for embeddings and chat completions).

When you see `import something as short_name`, it simply creates a nickname for a module so that later code is shorter and easier to read.

---

## 2. Loading the recipe dataset

```python
file_path = "_data/italian_recipes_clean.csv"
df = pd.read_csv(file_path)
print(df.info())
print(df.head())
```

- `file_path` stores where the CSV file lives. Storing paths in variables avoids retyping strings later.
- `pd.read_csv(...)` loads the CSV into a **DataFrame** called `df`. Think of a DataFrame as an Excel sheet inside Python.
- `df.info()` prints the structure: number of rows, column names, data types. Here we see 220 recipes with two text columns (`title` and `receipt`).
- `df.head()` prints the first five rows so we can sanity-check the data.

### Why do we need DataFrames?
Because we want to iterate over recipes, keep their titles, and eventually add new information (like embedding vectors) column by column. A DataFrame makes it easy to keep everything aligned.

---

## 3. What the system should do (natural language plan)

A markdown cell describes the goal:

> Take the user’s question → find the most similar 5 recipes → send them to ChatGPT → return a helpful answer.

This is the high-level outline of a RAG pipeline.

---

## 4. Building embeddings for similarity search

Embeddings are numerical representations (vectors) of text. Similar texts end up with vectors that are close to each other in high-dimensional space. We use these vectors to find which recipes best match the user’s request.

```python
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model_name = "text-embedding-3-small"
embeddings = []

for idx, row in df.iterrows():
    text = row["receipt"]
    if not isinstance(text, str) or text.strip() == "":
        embeddings.append(None)
        continue

    resp = client.embeddings.create(
        model=model_name,
        input=[text]
    )
    emb = resp.data[0].embedding
    embeddings.append(emb)

df["embedding"] = embeddings
```

Breakdown:

- `os.getenv("OPENAI_API_KEY")` looks for an environment variable named `OPENAI_API_KEY`. This keeps secrets out of the code. If it comes back `None`, the key is missing and API calls will fail.
- `client = OpenAI(...)` creates a reusable object for talking to OpenAI’s services.
- `model_name` picks the specific embedding model. `text-embedding-3-small` returns 1,536-number-long vectors.
- `df.iterrows()` lets us loop over each row in the DataFrame: `row` is a Series holding the columns for that recipe.
- We guard against empty or invalid text by appending `None` and skipping the embedding call.
- `client.embeddings.create(...)` sends the recipe text to the embedding endpoint. The API responds with a JSON payload; the actual vector lives at `resp.data[0].embedding`.
- At the end we attach the newly built list to a fresh DataFrame column `embedding` so each recipe row now carries its own vector.

Two sanity checks follow:

```python
type(df['embedding'][0])  # returns list
len(df['embedding'][0])   # returns 1536
```

This confirms that every stored embedding is a Python list of length 1536, as expected.

### Concept check: what is cosine similarity?
Cosine similarity measures how aligned two vectors are. If the angle between them is small (cosine close to 1), the texts are considered similar. The notebook even shows the mathematical formula later on.

---

## 5. Capturing the user’s question and embedding it

```python
user_text = """
Hi! I’d like to cook a good Italian dish for lunch! I have potatoes, carrots,
rosemary, and pork. Can you recommend a recipe and help me a bit with
preparation tips?
"""

resp = client.embeddings.create(
    model=model_name,
    input=[user_text]
)
user_query = resp.data[0].embedding
```

- A triple-quoted string preserves line breaks, making long prompts easier to read.
- We reuse the same embedding model so that recipe vectors and the user query live in the same vector space.
- `user_query` becomes a 1536-dimensional vector representing the user’s needs.

---

## 6. Finding the best matching recipes

```python
from scipy.spatial.distance import cosine

scores = []
for emb in df["embedding"]:
    if emb is None:
        scores.append(-1.0)
    else:
        scores.append(1.0 - cosine(np.array(emb), np.array(user_query)))

top5 = np.argsort(scores)[-5:]
```

Key ideas:

- `scipy.spatial.distance.cosine` computes **cosine distance** (0 means identical, 1 means opposite). To turn that into similarity, we subtract from 1.
- We loop over every recipe embedding. If an embedding is missing (`None`), we push a dummy score of `-1.0` so that it never ranks among the top results.
- `np.array(...)` converts Python lists to NumPy arrays. Many scientific functions expect NumPy arrays because they offer fast numerical operations.
- `np.argsort(scores)` returns the indices that would sort the scores. Taking `[-5:]` grabs the indices of the five highest scores.

Next, the code builds a string that holds the top recipe titles and their full instructions:

```python
output_lines = []
for i in top5:
    title = df.iloc[i]["title"]
    recipe = df.iloc[i]["receipt"]
    output_lines.append(f"{title}:\n{recipe}")

prompt_recipes = "\n\n".join(output_lines)
print(prompt_recipes)
```

- `df.iloc[i]` accesses the *i-th* row by position.
- We store each title + recipe text block in `output_lines` and join them with blank lines so that the language model later receives a readable chunk of examples.

The printed output lets us preview exactly which recipes were selected. In the sample run, recipes like “Loin of Pork Roasted” and “Vegetable Chowder” were highlighted—makes sense given the user’s ingredients.

The math cell right after this shows the cosine similarity formula once more, anchoring the concept with symbolic notation.

---

## 7. Prompting the language model with retrieved context

```python
prompt = f"""
You are a helpful Italian cooking assistant.
Here are some recipe examples I found that may or may not be relevant to the user's request:

{prompt_recipes}

User’s question: "{user_text}"

From the examples above:
1. Determine which recipes are relevant to what the user asked and which are not.
2. Discard or ignore irrelevant ones, and focus on relevant ones.
3. For each relevant example, rephrase the recipe in a narrative, conversational style, adding cooking tips, alternative ingredients, variations, or suggestions.
4. Then produce a final response to the user: a narrative that weaves together those enhanced recipes (titles + steps + tips) in an engaging way.
5. Don't forget to use the original titles of the recipes.
6. Advise on more than one recipe - if there are more than one relevant!

Do not just list recipes — tell a story, connect to the user's question, and use the examples as inspirations, but enhance them.
Make sure your response is clear, helpful, and focused on what the user wants.
"""
```

- This is an **f-string**, so `{prompt_recipes}` and `{user_text}` get replaced with their actual values.
- The prompt instructs the LLM on how to behave: decide relevance, enrich the recipes, and tell a coherent story. Clear instructions lead to better LLM outputs.

---

## 8. Generating the final response with ChatGPT

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful Italian cooking assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=1,
    max_tokens=5000
)

reply_text = response.choices[0].message.content
print(reply_text)
```

Explanation:

- `client.chat.completions.create(...)` hits the Chat Completions endpoint. The `messages` list sets the conversation: a **system message** defines the assistant’s personality, followed by the **user message** that includes context plus instructions.
- `temperature=1` allows for more creative responses (higher values mean more randomness; lower values are more deterministic).
- `max_tokens` controls how long the answer is allowed to be.
- The response JSON contains multiple choices; we pick the first one (`choices[0]`) and take its text content.
- Printing `reply_text` lets us read the final narrative. In the example run, the assistant recommended adapting “Loin of Pork Roasted” and “Minestrone alla Milanese” to match the user’s ingredients.

---

## 9. Putting it all together

1. **Data ingestion** with Pandas gives us a structured base of recipes.
2. **Embedding generation** transforms text into vectors that can be compared numerically.
3. **Similarity search** (via cosine similarity) picks the recipes most relevant to the user’s request.
4. **Prompt construction** packages the retrieved evidence plus instructions for the language model.
5. **LLM generation** returns a friendly, context-aware answer for the user.

This pattern—a retrieval step feeding into a generation step—is what makes RAG so powerful. Even this simple version already shows how to mix classical data processing (Pandas, NumPy, SciPy) with modern AI services (OpenAI embeddings and chat models). As you advance, you can replace the CSV with a vector database, add caching, validate outputs, or build a web app front-end. But Session 01 gives you the essential building blocks.

---

### Glossary for quick review
- **API key**: A secret token that authenticates your program with a web service.
- **DataFrame**: A 2D labeled table structure from `pandas` for data analysis.
- **Embedding**: A vector representation of text; similar meanings → similar vectors.
- **Cosine similarity**: A number between -1 and 1 indicating how similar two vectors are (1 means identical direction).
- **LLM (Large Language Model)**: A model like GPT-4 capable of generating human-like text based on input prompts.
- **Prompt**: The text instructions and context you send to an LLM to guide its response.

