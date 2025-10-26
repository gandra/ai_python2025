"""
GenAI/RAG in Python 2025 — Session 03 (script verzija)

Ovaj fajl je Python skripta koja radi isto što i Jupyter beležnica
"gandra/session03/session03.ipynb", ali je kod reorganizovan u jasne
funkcije sa detaljnim objašnjenjima i komentarima za učenje.

Teme koje pokriva:
- Postgres + pgvector kao vektorsko skladište (ingest + upiti sličnosti)
- Kreiranje embedinga korisničkog teksta (OpenAI embeddings API)
- Kontekstualizovan odgovor (Chat Completions)
- Function Calling primer (poziv „spoljne funkcije“ za vreme)

Napomena o izvršavanju:
- Ovaj skript pristupa mreži kada koristi OpenAI API i besplatni API za vreme.
  Za OpenAI deo potrebno je da je postavljena promenljiva okruženja
  OPENAI_API_KEY. Ako nije postavljena, skript će preskočiti te korake.

Kako instalirati zavisnosti (uv - preporučeno):
- Ako koristiš uv (brzi Python package manager):
    uv venv                   # kreira lokalni virtuelni env
    uv pip install -U pip     # opciono: osveži pip unutar venv-a
    uv add "psycopg[binary]" pgvector numpy pandas openai requests python-dotenv

  Objašnjenje:
  - "psycopg[binary]" je najlakši način za instalaciju psycopg v3 bez
    dodatnih sistemskih biblioteka (dolazi sa spakovanim binarnim delovima).
  - Paket „pgvector“ sadrži adapter za rad sa vektorima u Pythonu/psycopg.
  - Paket „openai" je novi v1+ SDK (from openai import OpenAI).

Docker Postgres sa pgvector ekstenzijom (lokalno):
- Preuzmi i pokreni kontejner:
    docker pull pgvector/pgvector:pg16
    docker run --name ragdb \
      -e POSTGRES_USER=raguser \
      -e POSTGRES_PASSWORD=ragpass \
      -e POSTGRES_DB=ragdb \
      -p 5432:5432 \
      -d pgvector/pgvector:pg16

- Provera i aktivacija ekstenzije u psql-u:
    docker exec -it ragdb psql -U raguser -d ragdb
    CREATE EXTENSION IF NOT EXISTS vector;
    SELECT 'pgvector ready' AS status;

Razlika: psycopg (v3) vs psycopg2 (v2) — SAŽETAK
- psycopg (ili „psycopg3“) je nova generacija drajvera za PostgreSQL u Python-u.
  Ključne prednosti:
  - Moderni API i bolja asinhrona podrška
  - Jednostavnije adaptere/ekstenzije (npr. pgvector)
  - Aktivno se razvija i preporučuje za nove projekte
- psycopg2 je starija, veoma stabilna v2 serija.
  - Ogroman ekosistem i mnogo primera
  - Nema iste modernizovane asink. mogućnosti kao v3
  - Prelazak v2 -> v3 uključuje male, ali bitne API razlike

Kada koji koristiti?
- Novi projekat: psycopg (v3) — preporuka, budućnost razvojne linije.
- Legacy kod koji već koristi psycopg2 i radi stabilno: ostati na psycopg2,
  osim ako aktivno refaktorišeš i želiš benefite v3.

Instalacija sa uv (kratko):
- psycopg v3 (preporuka):  uv add "psycopg[binary]"
- psycopg2 (ako ti baš treba v2): uv add psycopg2-binary
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import psycopg  # psycopg v3
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

# OpenAI v1+ SDK
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - opcija ako SDK nije instaliran
    OpenAI = None  # type: ignore


# -----------------------------
# Konfiguracija i konstante
# -----------------------------

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "_data", "italian_recipes_embedded.csv")


@dataclass
class DbConfig:
    """Konfiguracija za PostgreSQL konekciju.

    - host: adresa servera (npr. "localhost" kada radiš lokalno preko Dockera)
    - port: TCP port (podrazumevano 5432)
    - dbname: naziv baze
    - user: korisničko ime
    - password: lozinka

    Savet: U praksi čuvaj kredencijale u .env fajlu i učitaj ih iz okruženja.
    """

    host: str = "localhost"
    port: int = 5432
    dbname: str = "ragdb"
    user: str = "raguser"
    password: str = "ragpass"


# -----------------------------
# IO i transformacije podataka
# -----------------------------

def load_recipes_with_embeddings(csv_path: str = DATA_CSV) -> pd.DataFrame:
    """Učitaj dataset recepata sa već izračunatim embedding vektorima.

    Ovaj CSV (italian_recipes_embedded.csv) sadrži kolone: "title", "receipt",
    i "embedding" (string reprezentacija Python liste). Za rad sa pgvector,
    embedding moramo pretvoriti u numerički vektor (numpy array/list[float]).

    Vraća DataFrame i dodaje kolonu "embedding_vector" (np.ndarray float32).

    Ključna ideja: CSV čuva embedding kao string; ast.literal_eval + np.array
    konverzija pretvaraju ga nazad u listu/array brojeva. float32 štedi memoriju
    i odgovara većini modela/algoritama koji očekuju 32-bitne vektore.
    """

    df = pd.read_csv(csv_path)
    df["embedding_vector"] = df["embedding"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    return df


# -----------------------------
# Postgres + pgvector (ingest)
# -----------------------------

def connect_postgres(cfg: DbConfig) -> psycopg.Connection:
    """Napravi konekciju ka Postgres-u (psycopg v3) i registruj pgvector adapter.

    Zašto register_vector(conn)?
    - Omogućava da Python liste/np.arrays budu automatski mapirane na Postgres
      tip "vector" (pgvector ekstenzija), i obratno.
    """

    conn = psycopg.connect(
        host=cfg.host,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        port=cfg.port,
    )
    register_vector(conn)
    return conn


def reset_and_create_table(conn: psycopg.Connection, dim: int) -> None:
    """(Re)kreiraj tabelu `receipts` sa pgvector kolonom zadate dimenzije.

    - Bezbedno obriše prethodnu tabelu ako postoji (DROP TABLE IF EXISTS)
    - Napravi novu tabelu sa kolonom: embedding VECTOR(dim)

    Napomena: U produkciji je bolje imati migracije (npr. Alembic), a ne
    drop/create. Ovde je edukativni primer.
    """

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS receipts;")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS receipts (
            id SERIAL PRIMARY KEY,
            title   TEXT,
            receipt TEXT,
            embedding VECTOR({dim})
        );
        """
        cur.execute(create_sql)
        conn.commit()


def ingest_dataframe(conn: psycopg.Connection, df: pd.DataFrame) -> None:
    """Upiši sve redove iz DataFrame-a u tabelu `receipts`.

    Čuva: title, receipt, embedding (kao list[float]). Vrednost
    `df['embedding_vector']` pretvaramo u običnu Python listu (.tolist()) da bi
    je psycopg/pgvector adapter korektno poslao u Postgres.
    """

    insert_sql = """
        INSERT INTO receipts (title, receipt, embedding)
        VALUES (%s, %s, %s)
    """
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute(
                insert_sql,
                (row["title"], row["receipt"], row["embedding_vector"].tolist()),
            )
        conn.commit()


# -----------------------------
# OpenAI: embeddings i chat
# -----------------------------

def build_openai_client() -> Optional[OpenAI]:
    """Vrati OpenAI klijenta ako je API ključ dostupan, inače None.

    - OPENAI_API_KEY mora biti postavljen u okruženju.
    - Koristimo novi SDK (v1+): from openai import OpenAI; OpenAI(api_key=...).
    - Ako SDK nije instaliran ili ključ nedostaje, vrati None i preskoči pozive.
    """

    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def embed_text(client: OpenAI, text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Izračunaj embedding za dati tekst koristeći OpenAI embeddings API.

    - model: "text-embedding-3-small" je brz i pristupačan za demoe.
    - povratna vrednost: lista float vrednosti (vektor) pogodan za pgvector.
    """

    resp = client.embeddings.create(model=model, input=[text])
    return list(resp.data[0].embedding)


def query_similar_recipes(
    conn: psycopg.Connection, query_vector: Iterable[float], limit: int = 5
) -> pd.DataFrame:
    """Pokreni vektorski upit u Postgres/pgvector na osnovu unetog vektora.

    Koristimo operator udaljenosti <=> (cosine/prodizajn pgvector-a) i
    računamo sličnost kao 1 - distance. Rezultat vraćamo kao DataFrame.
    """

    sql = """
        SELECT id, title, receipt, 1 - (embedding <=> %s::vector) AS similarity
        FROM receipts
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(query_vector), list(query_vector), limit))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=colnames)


def build_cooking_prompt(user_text: str, prompt_recipes: pd.DataFrame) -> str:
    """Sastavi „rich" prompt koji ubacuje relevantne recepte kao kontekst.

    Ideja: model dobija primer(e) recepata koji su slični korisničkom upitu,
    a zatim dobija instrukciju da napravi narativni, prijateljski odgovor.
    """

    examples = prompt_recipes[["title", "receipt", "similarity"]].to_dict(orient="records")
    # Formatiraj kratko (može i JSON, ali čitljiv tekst je sasvim OK za LLM)
    formatted = []
    for ex in examples:
        formatted.append(
            f"Title: {ex['title']}\nSimilarity: {ex['similarity']:.3f}\nSteps: {ex['receipt']}\n"
        )
    prompt_recipes_text = "\n\n".join(formatted)

    prompt = f"""
You are a helpful Italian cooking assistant.

Here are some recipe examples that may or may not be relevant:

{prompt_recipes_text}

User’s question: "{user_text}"

From the examples above:
1) Identify relevant recipes and ignore the rest.
2) Rephrase relevant recipes in a friendly, narrative style with tips/variations.
3) Provide a final engaging answer. Use original recipe titles.
4) If multiple recipes are relevant, advise on more than one.
""".strip()
    return prompt


def chat_cooking_assistant(client: OpenAI, prompt: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
    """Pozovi Chat Completions API sa pripremljenim prompt-om.

    Vraća finalni tekst odgovora. Model možeš menjati po potrebi.
    """

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful Italian cooking assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ""


# -----------------------------
# Function Calling primer (vreme)
# -----------------------------

def get_current_weather(location: str) -> str:
    """Vrati kratki opis trenutnog vremena za dati grad koristeći wttr.in.

    Napomena: Ovo je edukativni primer „spoljne funkcije" koju model može da
    „pozove" (function calling). U realnosti ovde može biti bilo koji API ili
    tvoja poslovna logika. Nije potrebna API ključ za wttr.in, ali zahteva mrežu.
    """

    import requests  # lokalni import da izbegnemo nepotrebni dependency u testu

    url = f"http://wttr.in/{location}?format=j1"
    data = requests.get(url, timeout=15).json()
    current = data["current_condition"][0]
    temp_c = current["temp_C"]
    desc = current["weatherDesc"][0]["value"]
    return f"Temperature: {temp_c}°C, Condition: {desc}"


def run_function_calling_weather_demo(client: OpenAI) -> Optional[str]:
    """Mali demo kako Chat Completions bira i poziva alat (funkciju).

    Koraci:
    1) Modelu opišemo koji je alat dostupan (name/description/JSON schema).
    2) Model vrati tool_calls sa imenom funkcije i argumentima (kao JSON string).
    3) Naš kod pozove pravu Python funkciju i prenese rezultat nazad modelu.

    Vraća finalni, „obogaćen" odgovor modela ili None ako nešto nije dostupno.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city (Celsius only)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. Paris or London",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Paris right now?"}
    ]

    # 1) Model odlučuje da li i kako da pozove alat
    resp = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        return msg.content or ""

    # 2) Parsiramo argumente za pozvanu funkciju
    call = msg.tool_calls[0]
    args_json = call.function.arguments
    import json

    args = json.loads(args_json)
    location = args.get("location", "Paris")

    # 3) Pozovi Python funkciju i pošalji rezultat nazad modelu
    weather = get_current_weather(location)
    follow_up = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": messages[0]["content"]},
            msg,
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": weather,
            },
        ],
        temperature=0,
    )
    return follow_up.choices[0].message.content or ""


# -----------------------------
# Orkestracija (primer toka)
# -----------------------------

def run_demo_pipeline(user_text: Optional[str] = None, limit: int = 5) -> None:
    """End-to-end demo:

    1) Učitaj embeddings dataset i pripremi vektore
    2) Resetuj i kreiraj tabelu u Postgres-u, ingestuj podatke
    3) (Ako je OpenAI ključ dostupan) kreiraj embedding korisničkog teksta,
       pronađi slične recepte, izgradi prompt i pozovi chat
    4) (Opcionalno) Function calling demo (vreme)

    Ovaj metod ispisuje rezultate na stdout (print), jer je skripta edukativna.
    """

    df = load_recipes_with_embeddings(DATA_CSV)

    # 1) Postgres ingest
    cfg = DbConfig()
    conn = connect_postgres(cfg)
    try:
        dim = len(df["embedding_vector"][0])
        reset_and_create_table(conn, dim)
        ingest_dataframe(conn, df)

        # 2) OpenAI deo (ako postoji ključ)
        client = build_openai_client()
        if client is None:
            print("[info] OPENAI_API_KEY nije postavljen ili SDK nije instaliran — preskačem OpenAI korake.")
            return

        if not user_text:
            user_text = (
                "Hi! I’d like to cook a good Italian dish for lunch! "
                "I have potatoes, carrots, rosemary, and pork. Can you recommend a recipe and tips?"
            )

        user_vec = embed_text(client, user_text)
        prompt_recipes = query_similar_recipes(conn, user_vec, limit=limit)
        print("\nTop similar recipes (pgvector):\n", prompt_recipes)

        prompt = build_cooking_prompt(user_text, prompt_recipes)
        answer = chat_cooking_assistant(client, prompt)
        print("\nCooking assistant answer:\n", answer)

        # 3) Function calling demo (vreme)
        try:
            weather_ans = run_function_calling_weather_demo(client)
            if weather_ans:
                print("\nFunction-calling weather answer:\n", weather_ans)
        except Exception as e:
            # Edukativno: API može biti nedostupan iza firewalla/sandbox-a
            print(f"[warn] Function calling demo nije uspeo: {e}")

    finally:
        conn.close()


# -----------------------------
# Mini-„skripta za učenje": kratki sažetak pojmova
# -----------------------------

LEARNING_NOTES = """
Najvažnije ideje iz ovog primera:

- Vektorsko skladište: pgvector dodaje tip "vector" u Postgres i operatore
  za upite sličnosti (npr. <=>). Ovo omogućava semantičko pretraživanje
  direktno u SQL bazi.

- Embedding: pretvaramo tekst u niz brojeva (vektor) preko modela (OpenAI
  embeddings). Sličnost između dva teksta svodimo na sličnost između vektora.

- Ingest pipeline: učitavanje CSV -> konverzija string->vector -> insert u
  bazu -> indeksiranje/ upiti. U produkciji koristi migracije i bulk insert.

- Prompt kontekst: vraćene relevantne stavke (npr. recepti) dodaš u prompt da
  model može da ih "vidi" i napravi korisniji odgovor (RAG obrazac).

- Function Calling: model predlaže poziv "alata" (našeg Python koda) sa
  JSON argumentima. Mi izvršimo alat (API/računanje) i prosledimo rezultat
  nazad modelu da formira finalan odgovor.
""".strip()


def _print_notes() -> None:
    print("\nLEARNING NOTES:\n" + LEARNING_NOTES)


if __name__ == "__main__":
    # Pokreni end-to-end demo (ako je mreža/ključ dostupan) i odštampaj beleške
    try:
        run_demo_pipeline()
    except Exception as e:
        print(f"[warn] Demo pipeline nije uspeo ili je delimično preskočen: {e}")
    _print_notes()

