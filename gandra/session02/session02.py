"""Session 02 script verzija originalnog notbuka.

Skripta radi sve isto kao i notebook, ali uz dodatni korak: kada se prvi put
pokrene, ucitava recepte iz CSV fajla, racuna embedding vektore i cuva ih u
PostgreSQL bazi sa pgvector ekstenzijom. Svako sledece pokretanje koristi vec
snimljene vektore i preskace ponovno racunanje.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial import distance
from scipy.spatial.distance import cosine

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "psycopg2 is required to run this script. Install it with "
        "`pip install psycopg2-binary`."
    ) from exc

MODEL_NAME = "text-embedding-3-small"
EMBED_DIM = 1536
RECIPE_TABLE = "recipes"


# --- Utility helpers -------------------------------------------------------

def project_root() -> Path:
    """Vraca korenski direktorijum projekta koristeci apsolutnu putanju fajla."""

    return Path(__file__).resolve().parents[2]


def recipes_csv_path() -> Path:
    """Formira punu putanju do CSV fajla sa punim receptima."""

    return project_root() / "_data" / "italian_recipes_clean.csv"


def features_csv_path() -> Path:
    """Formira putanju do CSV fajla sa binarnim karakteristikama recepata."""

    return project_root() / "_data" / "italian_recipes_features.csv"


def make_openai_client() -> OpenAI:
    """Kreira OpenAI klijenta koristeci API kljuc iz okruzenja."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    return OpenAI(api_key=api_key)


def get_pg_connection():
    """Povezuje se na PostgreSQL bazu i koristi vrednosti iz docker-compose-a."""

    # Pokusavamo da procitamo promenljive iz okruzenja, a ako ih nema uzimamo
    # podrazumevane vrednosti iz docker-compose.yml (ai_rag korisnik/sifra).
    host = os.getenv("PGVECTOR_HOST") or os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("PGVECTOR_PORT") or os.getenv("POSTGRES_PORT") or "5422")
    database = (
        os.getenv("PGVECTOR_DB")
        or os.getenv("POSTGRES_DB")
        or "ai_rag"
    )
    user = (
        os.getenv("PGVECTOR_USER")
        or os.getenv("POSTGRES_USER")
        or "ai_rag"
    )
    password = (
        os.getenv("PGVECTOR_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or "ai_rag_password"
    )

    # psycopg2.connect otvara TCP vezu ka Postgres serveru.
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
    )
    conn.autocommit = True
    return conn


def ensure_pgvector_schema(conn) -> None:
    """Priprema bazu: pgvector ekstenzija, tabela i indeks za brze upite."""

    with conn.cursor() as cur:
        # Ekstenzija "vector" dodaje novi tip kolone koji cuva embedding vektore.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Tabela cuva id, naslov, tekst recepta i embedding vektor dimenzije 1536.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RECIPE_TABLE} (
                id BIGINT PRIMARY KEY,
                title TEXT NOT NULL,
                receipt TEXT NOT NULL,
                embedding VECTOR({EMBED_DIM})
            );
            """
        )
        # IVFFlat indeks ubrzava pretragu po vektorima koristeci cosine metriku.
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {RECIPE_TABLE}_embedding_cos_idx
              ON {RECIPE_TABLE} USING ivfflat (embedding vector_cosine_ops)
              WITH (lists = 100);
            """
        )


def table_row_count(conn) -> int:
    """Vraca broj redova u tabeli da znamo da li je baza vec popunjena."""

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {RECIPE_TABLE};")
        return cur.fetchone()[0]


def embed_texts(client: OpenAI, texts: Sequence[str], *, batch_size: int = 64) -> List[List[float]]:
    """Racuna embedding vektore za tekstove u manjim batch-evima radi stabilnosti."""

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        # Seckamo listu na manje delove da ne saljemo previse podataka odjednom.
        batch = list(texts[start : start + batch_size])
        if not batch:
            continue
        response = client.embeddings.create(model=MODEL_NAME, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def persist_recipes(conn, df: pd.DataFrame) -> None:
    """Upisuje recepte i njihove vektore u Postgres uz UPSERT logiku."""

    # reset_index() pravi nam stabilan numericki ID koji koristimo kao primarni kljuc.
    df_with_id = df.reset_index().rename(columns={"index": "id"})
    rows = []
    for _, row in df_with_id.iterrows():
        emb = row["embedding"]
        if emb is None:
            continue
        rows.append(
            (
                int(row["id"]),
                row["title"],
                row["receipt"],
                emb,
            )
        )
    if not rows:
        return
    with conn.cursor() as cur:
        # execute_values ubacuje vise redova odjednom; ON CONFLICT radi zamenu
        # ako smo isti ID ranije snimali (klasicni UPSERT).
        execute_values(
            cur,
            f"""
            INSERT INTO {RECIPE_TABLE} (id, title, receipt, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE
              SET title = EXCLUDED.title,
                  receipt = EXCLUDED.receipt,
                  embedding = EXCLUDED.embedding;
            """,
            rows,
        )


def vector_to_list(value) -> List[float] | None:
    """Normalizuje vrednost iz baze u obican Python list floatova."""

    if value is None:
        return None
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    # Kada psycopg2 vrati vektor kao string (npr. "[0.1,0.2,...]") rucno ga parsiramo.
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        items = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        return [float(item) for item in items]
    raise TypeError(f"Cannot convert value of type {type(value)} to list")


def to_pgvector_literal(values: Sequence[float]) -> str:
    """Pretvara listu floatova u string format `[a,b,...]` koji pgvector razume."""

    # pgvector očekuje decimalne brojeve u zagradi; formatiramo sa 8 decimala
    # da balansiramo preciznost i dužinu stringa.
    formatted = ",".join(f"{float(v):.8f}" for v in values)
    return f"[{formatted}]"


def load_recipes_from_db(conn) -> pd.DataFrame:
    """Ucitava sve recepte iz baze i konvertuje ih nazad u DataFrame."""

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, title, receipt, embedding FROM {RECIPE_TABLE} ORDER BY id;"
        )
        rows = cur.fetchall()
    records = []
    for rid, title, receipt, embedding in rows:
        records.append(
            {
                "id": int(rid),
                "title": title,
                "receipt": receipt,
                "embedding": vector_to_list(embedding),
            }
        )
    if not records:
        return pd.DataFrame(columns=["title", "receipt", "embedding"])
    df = pd.DataFrame(records).set_index("id")
    return df


def ensure_recipes_loaded(conn, client: OpenAI, df: pd.DataFrame) -> pd.DataFrame:
    """Vraca DataFrame sa embedding vektorima — iz baze ili sveze racunat."""

    ensure_pgvector_schema(conn)
    count = table_row_count(conn)
    if count:
        print(f"Found {count} recipes in {RECIPE_TABLE}, reusing stored embeddings.")
        return load_recipes_from_db(conn)

    # Ako tabela jos nema podatke, embedujemo CSV i snimamo rezultat.
    print("No recipes found in pgvector table; embedding CSV contents...")
    texts = df["receipt"].tolist()
    embeddings = embed_texts(client, texts)
    df = df.copy()
    df["embedding"] = embeddings
    persist_recipes(conn, df)
    print(f"Inserted {len(df)} recipes into {RECIPE_TABLE}.")
    return df


def query_top_k(conn, query_embedding: Sequence[float], *, limit: int = 5, probes: int = 10):
    """Vadi top K recepata direktno iz Postgresa koristeci pgvector cosine metriku."""

    with conn.cursor() as cur:
        # ivfflat.probes odredjuje koliko "listi" indeks pretrazuje (vise = tacnije).
        cur.execute(f"SET ivfflat.probes = {probes};")
        vector_literal = to_pgvector_literal(query_embedding)
        cur.execute(
            f"""
            SELECT id, title, receipt, 1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM {RECIPE_TABLE}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vector_literal, vector_literal, limit),
        )
        return cur.fetchall()


# --- Main pipeline --------------------------------------------------------

def main() -> None:
    """Glavna funkcija orkestrira ceo proces: data -> vektori -> preporuke."""

    client = make_openai_client()

    # 1) Ucitavamo CSV sa punim tekstom recepata da bismo imali osnovne podatke.
    recipes_df = pd.read_csv(recipes_csv_path())
    print(recipes_df.info())
    print(recipes_df.head())

    conn = get_pg_connection()
    try:
        # 2) Osiguravamo da su embedding vektori dostupni (iz baze ili novi).
        recipes_with_embeddings = ensure_recipes_loaded(conn, client, recipes_df)
        first_embedding = next(
            (emb for emb in recipes_with_embeddings["embedding"] if emb),
            None,
        )
        if first_embedding:
            print(type(first_embedding))
            print(len(first_embedding))

        # 3) Formiramo korisnicki upit i embedujemo ga jednim pozivom.
        user_text = (
            "Hi! I’d like to cook a good Italian dish for lunch! I have potatoes, carrots, "
            "rosemary, and pork. Can you recommend a recipe and help me a bit with "
            "preparation tips?"
        )
        user_query_embedding = embed_texts(client, [user_text])[0]
        print(type(user_query_embedding))
        print(len(user_query_embedding))

        # 4) Racunamo cosine slicnost lokalno (numpy) samo da pokazemo logiku.
        scores: List[float] = []
        for emb in recipes_with_embeddings["embedding"]:
            if emb is None:
                scores.append(-1.0)
            else:
                scores.append(1.0 - cosine(np.array(emb), np.array(user_query_embedding)))
        top5_idx = np.argsort(scores)[-5:][::-1]

        output_lines = []
        for idx in top5_idx:
            row = recipes_with_embeddings.iloc[idx]
            output_lines.append(f"{row['title'].strip()}:\n{row['receipt'].strip()}")
        prompt_recipes_local = "\n\n".join(output_lines)
        print(prompt_recipes_local)

        # 5) Istu stvar radimo direktno u bazi da pokazemo pgvector pretragu.
        db_hits = query_top_k(conn, user_query_embedding, limit=5, probes=10)
        prompt_recipes_db = "\n\n".join(
            f"{title.strip()}:\n{receipt.strip()}" for _, title, receipt, _ in db_hits
        )
        print(prompt_recipes_db)

        # 6) Kratak prikaz razlicitih mera udaljenosti/slicnosti.
        a = np.array([0.1, 0.3, 0.5, 0.0])
        b = np.array([0.2, 0.1, 0.4, 0.3])

        cos_sim = cosine(a, b)
        print("1. Cosine similarity:", cos_sim)
        euc_dist = distance.euclidean(a, b)
        print("2. Euclidean distance:", euc_dist)
        man_dist = distance.cityblock(a, b)
        print("3. Manhattan distance:", man_dist)

        a_bin = np.array([1, 0, 1, 0])
        b_bin = np.array([1, 1, 0, 0])
        jac_sim = 1 - distance.jaccard(a_bin, b_bin)
        print("Jaccard similarity:", jac_sim)

        # 7) Radimo i drugi pristup: binarne karakteristike + Jaccard.
        features_df = pd.read_csv(features_csv_path())
        print(features_df.info())

        user_request = """
Hey, I’m in the mood for something hearty but not too complicated. 
I’d love to cook a traditional Italian pasta dish, maybe with a rich tomato sauce, 
some garlic and olive oil, and a bit of Parmesan on top. 
I prefer something savory, not sweet — and ideally something that’s cooked on the stove, not baked. 
Any classic recipes you can recommend?
"""

        feature_prompt = f"""
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

Do not include explanations, extra text, or any other formatting—only the JSON object with keys and binary (0/1) values for each of the above features.

User request: {user_request}
"""

        feature_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful Italian cooking assistant."},
                {"role": "user", "content": feature_prompt},
            ],
            temperature=1,
            response_format={"type": "json_object"},
            max_tokens=5000,
        )

        tags = feature_response.choices[0].message.content
        print(tags)

        feature_dict = json.loads(tags)
        feature_cols = [col for col in features_df.columns if col not in {"title", "receipt"}]
        normalized_features = {
            key: int(feature_dict.get(key, 0)) for key in feature_cols
        }
        user_vector = pd.Series(normalized_features, index=feature_cols).astype(int)
        print(user_vector)

        feature_matrix = features_df[feature_cols].astype(int).values
        jaccard_similarities = [
            1 - distance.jaccard(user_vector, row) for row in feature_matrix
        ]
        features_df["jaccard_similarity"] = jaccard_similarities
        top5_features = features_df.nlargest(5, "jaccard_similarity")
        print(top5_features[["title", "receipt", "jaccard_similarity"]])

        prompt_recipes = ""
        for _, row in top5_features.iterrows():
            prompt_recipes += f"{row['title'].strip()}\n{row['receipt'].strip()}\n\n"
        prompt_recipes = prompt_recipes.strip()
        print(prompt_recipes)

        # 8) Konacno, pravimo prompt i menjamo temperaturu da vidimo kreativnost modela.
        final_prompt = f"""
You are a helpful Italian cooking assistant.  
Here are some recipe examples I found that may or may not be relevant to the user's request:

{prompt_recipes}

User’s question: "{user_request}"

From the examples above:
1. Determine which recipes are relevant to what the user asked and which are not.
2. Discard or ignore irrelevant ones, and focus on relevant ones.
3. For each relevant example, rephrase the recipe in a more narrative, conversational style, adding cooking tips, alternative ingredients, variations, or suggestions.
4. Then produce a final response to the user: a narrative that weaves together those enhanced recipes (titles + steps + tips) in an engaging way.
5. Don't forget to use the original titles of the recipes.
6. Advise on more than one recipe - if there are more than one relevant!

Do not just list recipes — tell a story, connect to the user's question, and use the examples as inspirations, but enhance them.  
Make sure your response is clear, helpful, and focused on what the user wants.
"""

        for temperature in [0.0, 1.5, 0.75, 1.25]:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful Italian cooking assistant."},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=temperature,
                max_tokens=5000,
            )
            reply_text = response.choices[0].message.content
            print(user_request)
            print(f"\n=== temperature: {temperature} ===\n")
            print(reply_text)

    finally:
        # Uvek zatvaramo konekciju da ne ostavimo otvorene resurse.
        conn.close()


if __name__ == "__main__":
    main()
