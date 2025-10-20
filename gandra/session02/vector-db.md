# CSV → Vector DB: Qdrant and PostgreSQL (pgvector)

This guide shows how to move from a CSV of recipes to a vector database, and then query it for similarity search. It covers two popular, open‑source options:

- Qdrant (standalone vector database)
- PostgreSQL with pgvector extension

The examples assume you already know how to build embeddings (see Session 01/02). We’ll reuse `_data/italian_recipes_clean.csv` with columns like `title`, `receipt`.

Prereqs
- Set `OPENAI_API_KEY` in your environment
- Install: `pip install pandas numpy openai qdrant-client psycopg2-binary` (plus `sqlalchemy` if preferred)
- Optional: Use the provided `docker-compose.yml` to run Qdrant and Postgres locally

---

## 1) Start services with Docker Compose

File: `gandra/session02/docker-compose.yml`
- Qdrant: REST on `http://localhost:6333`
- Postgres (with pgvector): `postgres://postgres:postgres@localhost:5432/recipes`

Start and check:
- `docker compose -f gandra/session02/docker-compose.yml up -d`
- Qdrant UI: open `http://localhost:6333/dashboard`
- Postgres psql shell: `docker exec -it session02-postgres psql -U postgres -d recipes`

---

## 2) Shared embedding utility (Python)

Generate recipe embeddings with OpenAI once, then reuse for either DB.

```python
import os
import pandas as pd
from openai import OpenAI

MODEL = "text-embedding-3-small"  # 1536 dims

def embed_texts(texts):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Batch in chunks if needed; here we keep it simple
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in resp.data]

def load_and_embed(csv_path):
    df = pd.read_csv(csv_path)
    texts = [t if isinstance(t, str) and t.strip() else "" for t in df["receipt"]]
    embs = embed_texts(texts)
    df["embedding"] = embs
    return df

# Example usage
# df = load_and_embed("_data/italian_recipes_clean.csv")
```

Tips
- For large datasets, chunk `input=texts` to avoid payload limits.
- Cache `df[[title, receipt, embedding]]` (e.g., Parquet) so you don’t re‑bill embeddings each run.

---

## 3) Qdrant

Qdrant is a high‑performance, open‑source vector DB with a simple API.

Install client
- `pip install qdrant-client`

Create collection and upsert points

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

QDRANT_URL = "http://localhost:6333"
COLLECTION = "recipes_qdrant"
DIM = 1536  # embedding size for text-embedding-3-small

client = QdrantClient(url=QDRANT_URL)

# Create collection (idempotent)
if COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

# Suppose df has columns: title, receipt, embedding (list[float] len=1536)
points = []
for idx, row in df.iterrows():
    if not isinstance(row["embedding"], list) or not row["embedding"]:
        continue
    points.append(
        PointStruct(
            id=int(idx),
            vector=row["embedding"],
            payload={
                "title": row["title"],
                "receipt": row["receipt"],
            },
        )
    )

if points:
    client.upsert(collection_name=COLLECTION, points=points)
```

Similarity search (Top‑K)

```python
query = "I have potatoes, carrots, rosemary, and pork; suggest an Italian lunch"
q_emb = embed_texts([query])[0]

hits = client.search(
    collection_name=COLLECTION,
    query_vector=q_emb,
    limit=5,
)

for h in hits:
    print(h.id, h.score, h.payload.get("title"))
```

Filtering (optional)
- Add payload filters, e.g., only recipes with certain tags.
- See `qdrant-client` docs for `Filter`, `FieldCondition`, `Match*`.

---

## 4) PostgreSQL + pgvector

This uses Postgres with the pgvector extension for vector columns and ANN indexes.

Connect and enable pgvector

```sql
-- In psql connected to DB 'recipes'
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for recipes
DROP TABLE IF EXISTS recipes;
CREATE TABLE recipes (
  id        BIGINT PRIMARY KEY,
  title     TEXT,
  receipt   TEXT,
  embedding VECTOR(1536)  -- must match model dimension
);

-- Recommended index (cosine)
-- Requires pgvector >= 0.5.0 for cosine ops
CREATE INDEX IF NOT EXISTS recipes_embedding_cos_idx
  ON recipes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- For better recall/latency, tune lists (and later probes per session)
```

Ingest rows from Python

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="recipes",
    user="postgres", password="postgres"
)
conn.autocommit = True

with conn, conn.cursor() as cur:
    # Upsert rows (simple version: delete then insert)
    cur.execute("DELETE FROM recipes;")
    for idx, row in df.iterrows():
        emb = row["embedding"]
        if not isinstance(emb, list) or not emb:
            continue
        # pgvector accepts arrays as Python lists; cast to vector via %s
        cur.execute(
            """
            INSERT INTO recipes (id, title, receipt, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
              SET title = EXCLUDED.title,
                  receipt = EXCLUDED.receipt,
                  embedding = EXCLUDED.embedding;
            """,
            (int(idx), row["title"], row["receipt"], emb)
        )
```

Similarity search (cosine)

```python
with conn, conn.cursor() as cur:
    q_emb = embed_texts([query])[0]
    # <=> is cosine distance in pgvector; lower is better
    cur.execute(
        """
        SELECT id, title, 1 - (embedding <=> %s) AS cosine_similarity
        FROM recipes
        ORDER BY embedding <=> %s  -- ascending distance
        LIMIT 5;
        """,
        (q_emb, q_emb)
    )
    for rid, title, sim in cur.fetchall():
        print(rid, round(sim, 4), title)
```

Index/ANN tips
- Build `ivfflat` index after loading data; for large loads, drop index → bulk insert → create index.
- Set session `SET ivfflat.probes = 10;` (or higher) to increase recall at some latency cost.
- If using L2 distance instead of cosine, use `vector_l2_ops` and `<->` operator.

---

## 5) Data modeling and chunking

Text chunking
- Long documents benefit from splitting into smaller chunks (e.g., 500–1000 tokens) with overlap. Store each chunk as a separate vector with a `doc_id`/`chunk_id` payload.

Metadata
- Store `title`, `source`, `url`, `ingredients`, or tags in payload/columns. Use filters alongside vector search to refine results.

Consistency
- The embedding model dimension (e.g., 1536) must match collection/table definition. If you switch models, re‑index.

---

## 6) Putting it together (mini‑pipeline)

```python
from pathlib import Path

CSV = "_data/italian_recipes_clean.csv"
df = load_and_embed(CSV)

# Choose one backend:
use_qdrant = True

if use_qdrant:
    # Create collection and upsert
    # (reuse code from Qdrant section)
    pass
else:
    # Create table, index, and insert
    # (reuse code from Postgres section)
    pass

# Query
query = "Traditional Italian pasta with tomato, garlic, olive oil"
# (see the Qdrant or Postgres query code above)
```

---

## 7) Troubleshooting

- Embedding errors: ensure `OPENAI_API_KEY` and internet connectivity.
- Dimension mismatch: Qdrant size or Postgres `VECTOR(…)` must match your model.
- Performance: for Postgres, tune `lists` and `probes`; for Qdrant, review HNSW/quantization settings if needed.
- Persistence: Qdrant stores data under the Docker volume; Postgres under `pgdata` volume.

