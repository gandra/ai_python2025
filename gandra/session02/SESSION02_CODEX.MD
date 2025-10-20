# Session 02 — Expanding the Basic RAG Framework (Walkthrough)

This session builds on Session 01 and takes you through a richer Retrieval-Augmented Generation (RAG) workflow. We still retrieve relevant recipes and prompt an LLM, but now we also:

- Compare multiple similarity/distance metrics (cosine, Euclidean, Manhattan, Jaccard)
- Work with a dataset of discrete “features” and match user intent to recipes using Jaccard similarity
- Explore LLM “determinism” via `temperature` and `top_p`

Everything below explains each code section step-by-step and tells you what to do to run it successfully.

---

## 1. Imports and setup

```python
import os
import numpy as np
import pandas as pd
from openai import OpenAI
```

- `os`: reading env vars like `OPENAI_API_KEY`.
- `numpy`: vector math (fast numerical arrays).
- `pandas`: `DataFrame` tables for CSV data.
- `OpenAI`: Python SDK client to call embeddings and chat models.

What you need to do:
- Ensure `OPENAI_API_KEY` is set in your shell before running the notebook:
  - macOS/Linux: `export OPENAI_API_KEY=sk-...`
  - Windows (PowerShell): `$Env:OPENAI_API_KEY = "sk-..."`
- Install dependencies (recommend): `pip install -r requirements.txt` or `pip install numpy pandas scipy openai jupyter`

---

## 2. Load the recipes dataset

```python
file_path = "_data/italian_recipes_clean.csv"
df = pd.read_csv(file_path)
print(df.info())
print(df.head())
```

- Loads a CSV of Italian recipes with columns like `title` and `receipt`.
- `df.info()` shows schema; `df.head()` shows the first few rows.

What you need to check:
- Make sure the file path exists relative to your current working directory.

Optional quick peeks:

```python
df
df["receipt"][0]
```

These display the full DataFrame and the raw text of the first recipe.

---

## 3. Build vector embeddings for similarity search

Embeddings turn text into vectors. Similar recipes end up with vectors that are close together. We use an OpenAI embedding model.

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

    # Call the embeddings endpoint
    resp = client.embeddings.create(
        model=model_name,
        input=[text]
    )
    emb = resp.data[0].embedding
    embeddings.append(emb)

df["embedding"] = embeddings
```

Notes:
- The notebook contains a TODO (“Call the embeddings”). Use the exact call shown above to complete it, then assign `df["embedding"] = embeddings`.
- Some rows may have empty text; we store `None` to skip them later.

### Detailed (exact steps to fix the TODO):

TODO...

Quick checks:

```python
type(df['embedding'][0])   # list
len(df['embedding'][0])    # 1536 for text-embedding-3-small
```

---

## 4. User input and its embedding

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

print(type(user_query))
print(len(user_query))
```

- We embed the user’s request with the same model so it lives in the same vector space as recipe embeddings.

---

## 5. Cosine similarity retrieval (Top‑K)

```python
from scipy.spatial.distance import cosine

scores = []
for emb in df["embedding"]:
    if emb is None:
        scores.append(-1.0)
    else:
        scores.append(1.0 - cosine(np.array(emb), np.array(user_query)))

top5_idx = np.argsort(scores)[-5:]  # indices of 5 highest scores

# Build prompt context from those rows
output_lines = []
for i in top5_idx:
    title = df.iloc[i]["title"]
    recipe = df.iloc[i]["receipt"]
    output_lines.append(f"{title}:\n{recipe}")

prompt_recipes = "\n\n".join(output_lines)
print(prompt_recipes)
```

Why “1 − cosine distance”? `scipy.spatial.distance.cosine(a, b)` returns a distance (0 = identical). We want similarity (1 = identical), so we compute `1 − distance`.

Math reminder:

cos(θ) = (a · b) / (||a|| · ||b||)

---

## 6. More distance/similarity metrics

We explore several common metrics using small example vectors.

```python
from scipy.spatial import distance

a = np.array([0.1, 0.3, 0.5, 0.0])
b = np.array([0.2, 0.1, 0.4, 0.3])

# Cosine (distance; convert to similarity via 1 - d)
cos_dist = distance.cosine(a, b)
cos_sim  = 1 - cos_dist

# Euclidean (L2)
euc_dist = distance.euclidean(a, b)

# Manhattan / Cityblock (L1)
man_dist = distance.cityblock(a, b)

print("cos_dist:", cos_dist, "cos_sim:", cos_sim)
print("euc_dist:", euc_dist)
print("man_dist:", man_dist)
```

Core ideas:
- Cosine focuses on direction (alignment), ignoring vector length.
- Euclidean is straight-line distance in n‑dim space.
- Manhattan sums absolute per‑dimension differences (grid distance).

Jaccard is used for sets/binary features:

similarity_J(A,B) = |A ∩ B| / |A ∪ B|

distance_J(A,B) = 1 − similarity

---

## 7. Discrete features pipeline (Jaccard matching)

We switch to a different dataset with binary features for each recipe.

```python
file_path = "_data/italian_recipes_features.csv"
df = pd.read_csv(file_path)
print(df.info())
```

Example user intent:

```python
user_request = """
Hey, I’m in the mood for something hearty but not too complicated.
I’d love to cook a traditional Italian pasta dish, maybe with a rich tomato sauce,
some garlic and olive oil, and a bit of Parmesan on top.
I prefer something savory, not sweet — and ideally something that’s cooked on the stove, not baked.
Any classic recipes you can recommend?
"""
```

Tag the request with an LLM to produce a JSON of 0/1 features:

```python
prompt = f"""
You are given a user's request.
Based on the request, output ONLY a valid JSON object with the following binary features,
where each value must be either 0 or 1:

[
  "is_soup_broth", "is_pasta", "is_rice", "is_meat_dish", "is_fish_dish", "is_egg_dish",
  "is_vegetable_dish", "is_dessert", "contains_pasta", "contains_rice", "contains_meat",
  "contains_fish_seafood", "contains_egg", "contains_cheese", "contains_tomato",
  "contains_olive_oil", "contains_garlic", "contains_wine", "contains_herbs",
  "is_boiled", "is_baked", "is_fried", "is_grilled", "is_raw_preparation",
  "is_sauce_based", "is_slow_cooked", "has_stuffing", "served_with_sauce", "is_soup_like",
  "is_bread_based", "is_spicy", "is_savory", "is_sweet", "contains_citrus",
  "mentions_region", "mentions_dialect_term", "is_classic_named_dish"
]

Do not include explanations or extra text—only JSON.

User request: {user_request}
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Italian cooking assistant."},
        {"role": "user",   "content": prompt}
    ],
    temperature=1,
    response_format={"type": "json_object"},  # force JSON
    max_tokens=5000
)

tags = response.choices[0].message.content
```

Parse JSON, convert to a vector, compute Jaccard similarity against the dataset:

```python
import json
feature_dict  = json.loads(tags)
feature_cols  = list(feature_dict.keys())
user_vector   = pd.Series(feature_dict, index=feature_cols).astype(int)

feature_matrix = df[feature_cols].astype(int).values

from scipy.spatial import distance
jaccard_similarities = [1 - distance.jaccard(user_vector, row) for row in feature_matrix]

df['jaccard_similarity'] = jaccard_similarities
top5_df = df.nlargest(5, 'jaccard_similarity')
print(top5_df[['title', 'receipt', 'jaccard_similarity']])
```

Build prompt context from the top 5 recipes:

```python
prompt_recipes = ""
for _, row in top5_df.iterrows():
    prompt_recipes += f"{row['title'].strip()}\n"
    prompt_recipes += f"{row['receipt'].strip()}\n\n"
prompt_recipes = prompt_recipes.strip()
```

Tip: In the notebook, `top5` is used both for cosine indices and later as a DataFrame. To avoid confusion, use different names like `top5_idx` (indices) and `top5_df` (DataFrame).

---

## 7b. Jaccard Similarity — Deep Dive

What it measures (beginner‑friendly):
- Think of each recipe and the user’s wish list as checklists (binary features 0/1).
- Jaccard similarity tells you “what fraction of the combined checklist is actually shared”. If both want pasta and tomato, that’s overlap; if the recipe has cheese but the user didn’t ask for it, that is counted only in the union (not the overlap).

Set definition (classic):
- For sets A and B:  similarity_J(A, B) = |A ∩ B| / |A ∪ B|
- Distance is 1 − similarity. SciPy’s `distance.jaccard(u, v)` returns the distance, not the similarity.

Binary‑vector view (what we use here):
- Treat 0/1 vectors. Overlap is positions where both have 1. Union is positions where at least one has 1.

Small example (intuition):
```
user = [1, 1, 1, 0]        # pasta, tomato, cheese, not baked
recA = [1, 1, 1, 0]        # perfect match → overlap=3, union=3 → sim=1.0
recB = [1, 1, 0, 1]        # overlap=2 (pasta,tomato), union=4 → sim=0.5
```

SciPy behavior and gotchas:
- `distance.jaccard(u, v)` expects 1D arrays of equal length; non‑binary values are allowed but interpret carefully.
- For binary use‑cases, ensure ints 0/1: `astype(int)` and clamp values if needed.
- If both vectors are all zeros (no features set), SciPy returns distance 0.0 (they are identical), so similarity = 1 − 0 = 1.0. In practice, you might want to special‑case “all zeros” as “no information” and treat similarity as 0.0 — pick the convention that matches your product logic.

Manual implementation (for clarity):
```python
import numpy as np

def jaccard_similarity_binary(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    intersection = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    return 1.0 if union == 0 else intersection / union

# Compare with SciPy
from scipy.spatial import distance
def jaccard_similarity_scipy(a, b):
    return 1 - distance.jaccard(a, b)
```

Weighted Jaccard (when features have strengths, not just 0/1):
- If you have non‑negative counts or weights per feature, a common variant is:
  similarity_wJ(a, b) = sum(min(a_i, b_i)) / sum(max(a_i, b_i))
- This down‑weights mismatches and rewards overlap where both have larger values.

Practical pipeline tips (to avoid bugs):
- Align columns: ensure `feature_cols` in the user vector and DataFrame are in the same order. Use a single `expected` list as the source of truth.
- Validate keys from LLM: fill missing keys with 0; coerce values to 0/1.
- Keep types tidy: `df[feature_cols].astype(int).values` for the matrix; `user_vector.astype(int)`.
- Handle “ties”: when several recipes tie on similarity, break ties by a secondary score (e.g., number of overlapping features, or cosine score if available).
- Thresholding: optionally filter out low Jaccard matches (e.g., < 0.2) before prompting the LLM.

End‑to‑end (the discrete track):
1) LLM turns free‑text request into a binary feature JSON.  
2) Parse → validate → build `user_vector`.  
3) Compute Jaccard similarity vs. every recipe (row in `feature_matrix`).  
4) Pick Top‑K (e.g., 5).  
5) Build `prompt_recipes` from those rows and send to the chat model.

Naming clarity:
- Use `top5_idx` for cosine (indices), and `top5_df` for Jaccard (`df.nlargest(...)` returns a DataFrame). This avoids confusion later when iterating (`for i in top5_idx` vs `for _, row in top5_df.iterrows()`).

Testing yourself (quick sanity):
- Create two tiny binary vectors and compute both the manual and SciPy versions. Confirm they match. Then try corner cases (all zeros; no overlap; complete overlap).

---

## 8. Prompting the LLM and exploring determinism

We create a user-facing prompt that includes the retrieved recipes. Then we call the Chat Completions API several times with different `temperature` values to see how the text changes.

```python
prompt = f"""
You are a helpful Italian cooking assistant.
Here are some recipe examples I found that may or may not be relevant to the user's request:

{prompt_recipes}

User’s question: "{user_request}"

From the examples above:
1. Determine which recipes are relevant and which are not.
2. Ignore irrelevant ones and focus on relevant ones.
3. For each relevant example, rephrase it in a narrative style, adding tips and variations.
4. Produce a final, engaging response that weaves them together.
5. Use the original recipe titles.
6. Recommend more than one recipe if appropriate.
"""
```

Generate responses with different temperatures:

```python
for T in [0.0, 0.75, 1.25, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful Italian cooking assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=T,
        max_tokens=5000
    )
    print("\n=== temperature:", T, "===\n")
    print(response.choices[0].message.content)
```

Definitions:
- `temperature`: controls randomness. Lower → more deterministic; higher → more creative.
- `top_p` (nucleus sampling): samples only from the smallest set of tokens whose cumulative probability ≥ `top_p`. Lower `top_p` narrows the token pool; `top_p=1.0` uses the full distribution.

---

## 9. End‑to‑end summary (what happens overall)

1. Load recipes from CSV into a DataFrame.
2. Build embeddings for each recipe (`text-embedding-3-small`).
3. Embed the user query the same way.
4. Retrieve top recipes with cosine similarity, or (in the discrete feature pipeline) with Jaccard.
5. Construct a prompt that includes retrieved recipes and clear instructions.
6. Call a chat model to produce a helpful, context-aware response.
7. Adjust `temperature`/`top_p` to trade off determinism vs. creativity.

---

## 10. Do-this checklist (exact steps to run)

- Set your API key in the environment: `export OPENAI_API_KEY=sk-...`.
- Install packages: `pip install numpy pandas scipy openai jupyter` (or `pip install -r requirements.txt`).
- Start Jupyter: `jupyter lab` (or `jupyter notebook`).
- Open `gandra/session02.ipynb`.
- Run cells in order. Where you see the TODO “Call the embeddings”, paste the embedding call block from Section 3.
- Ensure the data files exist: `_data/italian_recipes_clean.csv` and `_data/italian_recipes_features.csv`.
- If you hit variable name confusion around `top5`, rename as suggested (`top5_idx`, `top5_df`).
- For tagging (JSON), keep `response_format={"type": "json_object"}` to force valid JSON.

---

## 11. Troubleshooting

- Missing API key → `OpenAIAuthenticationError` or similar. Set `OPENAI_API_KEY`.
- Network/API issues → try again or check your internet connection and model availability.
- Path errors → confirm `_data/...` paths relative to your working directory.
- `None` embeddings → skip those rows when scoring; see the guard in Section 3.
- Shape/dtype issues → wrap lists with `np.array(...)` before calling `scipy` functions.

That’s it — Session 02 gives you two retrieval tracks (vector vs. features) and shows how LLM sampling parameters affect responses. This is the foundation for more advanced RAG systems.
