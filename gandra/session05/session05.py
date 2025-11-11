"""
GenAI/RAG in Python 2025 — Session 05 (script verzija)

Ovaj fajl je Python skripta koja radi isto što i Jupyter beležnica
"gandra/session05/session05.ipynb", ali je kod reorganizovan u jasne
funkcije sa detaljnim objašnjenjima i komentarima za učenje.

Tema: spajanje internog RAG-a (vektori recepata) sa Google Programmable
Search Engine (PSE) — kada je interni kontekst slab, agent po potrebi zove
web pretragu (Custom Search JSON API) i zatim priprema finalni odgovor.

Preuslovi (preporučeno):
- Postavi OPENAI_API_KEY u okruženju
- Postavi GOOGLE_CSE_API_KEY i GOOGLE_CSE_CX (Search engine ID)
- `uv sync`, zatim pokretanje: `uv run python gandra/session05/session05.py`
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cosine

# OpenAI v1+ SDK
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Konstante i putanje
# -----------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL_PLAN = os.getenv("CHAT_MODEL_PLAN", "gpt-4")
CHAT_MODEL_EXEC = os.getenv("CHAT_MODEL_EXEC", "gpt-4")

DATA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "_data", "italian_recipes_embedded.csv"
)


def load_env_if_available() -> None:
    """(Opcionalno) učitaj .env ako je python-dotenv instaliran.

    Starter savet: .env fajl je praktičan za lokalni razvoj. Na serverima i CI/CD
    je bolje koristiti secrets/varijable okruženja.
    """

    try:  # pragma: no cover
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def build_openai_client() -> Optional[OpenAI]:
    """Vrati OpenAI klijenta ako je API ključ dostupan, inače None.

    Za vežbu: uvek proveravaj pre nego što pozoveš mrežni API. Nemoj rušiti skriptu
    ako ključ nije postavljen — edukativni primeri mogu tada preskočiti taj deo.
    """

    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def google_search(
    query: str,
    num: int = 5,
    *,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    timeout: int = 20,
) -> List[Dict[str, str]]:
    """Pozovi Google Custom Search JSON API i vrati listu rezultata.

    - query: tekst upita
    - num: koliko rezultata (1..10)
    - api_key/cx: preuzmi iz argumenata ili iz okruženja (GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX)

    Vraća listu elemenata: {title, link, snippet}.
    """

    api_key = api_key or os.getenv("GOOGLE_CSE_API_KEY")
    cx = cx or os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cx:
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": max(1, min(int(num), 10)),  # API cap num<=10
        "safe": os.getenv("GOOGLE_SAFE", "active"),  # active|off
        "lr": os.getenv("GOOGLE_LR", ""),  # npr. lang_en
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", []) or []
    return [
        {
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "snippet": it.get("snippet", ""),
        }
        for it in items
    ]


def load_embeddings_df(csv_path: str = DATA_CSV) -> pd.DataFrame:
    """Učitaj CSV i parsiraj kolonu "embedding" u np.ndarray(float32).

    Ključna ideja: CSV čuva embedding kao string (npr. "[0.1, -0.2, ...]").
    Funkcija pretvara u numerički vektor pogodan za proračune kosinusne sličnosti.
    """

    df = pd.read_csv(csv_path)
    df["embedding_vector"] = df["embedding"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    return df


def embed_query(client: OpenAI, text: str) -> List[float]:
    """Izračunaj embedding korisničkog teksta jednim pozivom.

    Zašto embedding? RAG upoređuje semantičku sličnost vektora umesto prostog
    sintaksičkog poklapanja reči.
    """

    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return list(resp.data[0].embedding)


def rag_retrieve(user_vec: Iterable[float], df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Vrati top-k naj-sličnijih recepata prema kosinusnoj sličnosti (1 - cosine).

    - user_vec: embedding korisničkog upita
    - df: DataFrame sa kolonom "embedding_vector"
    - rezultat: kolone id, title, receipt, similarity
    """

    user = np.array(list(user_vec), dtype=np.float32)
    sims: List[float] = []
    for _, row in df.iterrows():
        emb = row["embedding_vector"]
        sim = 1.0 - float(cosine(user, emb)) if isinstance(emb, np.ndarray) else -1.0
        sims.append(sim)

    df = df.copy()
    df["similarity"] = sims
    top = df.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)
    return pd.DataFrame(
        {
            "id": top.index,
            "title": top["title"],
            "receipt": top["receipt"],
            "similarity": top["similarity"],
        }
    )


def build_tools_spec() -> List[Dict[str, Any]]:
    """Specifikacija alata za Chat Completions (edukativno, single tool: google_search)."""

    return [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web for Italian cuisine info when RAG is insufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to send to Google"},
                        "num": {
                            "type": "integer",
                            "description": "How many results (1..10)",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def plan_need_search(client: OpenAI, user_prompt: str, rag_df: pd.DataFrame) -> Dict[str, Any]:
    """Zamoli model da napravi plan i odluči da li treba web pretraga.

    Vraća JSON: {need_search: bool, search_query: str, rationale: str, plan: str}
    """

    instruction = (
        "You are a planning assistant. Decide if web search is needed to improve answer quality "
        "for the provided user question. "
        "Return JSON with fields: need_search (true/false), search_query (string), rationale (string), "
        "and then propose a short step-by-step plan for composing the final answer. "
        "The RAG context needs to include five (5) specific recipes closely matching the ingredients."
    )
    context = "\n\n".join(rag_df["receipt"].astype(str).tolist())
    prompt = (
        instruction
        + f"\n\n### USER QUESTION ###: {user_prompt}\n\n"
        + f"### RAG CONTEXT ###:\n{context}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL_PLAN,
        messages=[{"role": "user", "content": prompt}],
        tools=build_tools_spec(),
        temperature=0,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        # Ako model ne vrati validan JSON, vrati konzervativni plan bez pretrage.
        return {"need_search": False, "search_query": "", "rationale": text, "plan": "rag-only"}


def render_web_context(items: List[Dict[str, str]]) -> str:
    """Pretvori listu Google rezultata u čitljiv tekstualni kontekst za prompt."""

    parts = []
    for it in items:
        parts.append(
            f"Title: {it.get('title','')}\nLink: {it.get('link','')}\nSnippet: {it.get('snippet','')}\n"
        )
    return "\n".join(parts)


def build_self_prompt(user_prompt: str, rag_context: str, web_context: str) -> str:
    """Napiši prompt za „drugi“ LLM — koristimo self-prompting pristup.

    Važan detalj: koristimo placeholders {user_prompt}, {rag_context}, {web_context}
    i zatim .format(...) ubacujemo konkretne vrednosti.
    """

    instruction = (
        "You are a prompt engineer. Compose the best possible prompt for a Large Language Model (LLM) "
        "to answer the provided user question. The RAG CONTEXT provides results from a vector search, "
        "and the WEB CONTEXT may augment it with web links/snippets. Do not answer; return only the prompt text. "
        "Be systematic, add sections, and precise instructions. Begin your prompt with: The user is asking"
    )
    return (
        instruction
        + "\n\n### OUTPUT FORMAT ###\n"
        + "- A plain string that instructs another LLM and answers the user question,\n"
        + "- always using variables {user_prompt}, {web_context}, {rag_context} (curly braces),\n"
        + "- always beginning with the words: The user is asking\n"
        + "\n"
        + "Use these variables directly in your output string: {user_prompt}, {web_context}, {rag_context}."
    )


def run_agent(user_prompt: str) -> str:
    """Kratki agent: RAG -> plan -> opcioni Google -> self-prompt -> konačan odgovor.

    Vraća finalni tekst odgovora ili informaciju da su preskočeni mrežni koraci.
    """

    load_env_if_available()
    client = build_openai_client()
    if client is None:
        return "[info] OPENAI_API_KEY nije postavljen ili SDK nije instaliran — preskačem OpenAI korake."

    # 1) Učitavanje embedding dataset-a i RAG retrieval
    df = load_embeddings_df(DATA_CSV)
    user_vec = embed_query(client, user_prompt)
    rag = rag_retrieve(user_vec, df, top_k=5)

    # 2) Plan — da li treba web pretraga?
    plan = plan_need_search(client, user_prompt, rag)

    # 3) Ako treba, pozovi Google i pripremi web kontekst
    web_items: List[Dict[str, str]] = []
    if plan.get("need_search"):
        q = str(plan.get("search_query") or user_prompt)
        web_items = google_search(q, num=10)

    rag_context = "\n\n".join(rag["receipt"].astype(str).tolist())
    web_context = render_web_context(web_items)

    # 4) Self-prompt: model napiše instrukcije za „drugog“ modela
    self_prompt = build_self_prompt(user_prompt, rag_context, web_context)
    resp = client.chat.completions.create(
        model=CHAT_MODEL_PLAN,
        messages=[{"role": "user", "content": self_prompt}],
        temperature=0,
    )
    templated_prompt = resp.choices[0].message.content or ""

    # 5) Umetni konkretne vrednosti u šablon i izvrši završni upit
    final_prompt = templated_prompt.format(
        user_prompt=user_prompt, rag_context=rag_context, web_context=web_context
    )
    final = client.chat.completions.create(
        model=CHAT_MODEL_EXEC,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0,
    )
    return final.choices[0].message.content or ""


# -----------------------------
# Mini-„skripta za učenje": kratki sažetak pojmova
# -----------------------------

LEARNING_NOTES = """
Najvažnije ideje iz ovog primera:

- PSE (Programmable Search Engine) + Custom Search API: način da agent, kada je
  RAG kontekst slab, povuče svež web signal (naslovi/linkovi/snippeti).
- RAG retrieval (kosinusna sličnost): vektorska sličnost često daje bolje
  rezultate od ključnih reči; integracija sa embedding modelom je ključna.
- Planiranje -> Alati -> Self-prompting: jednostavan obrazac za izgradnju
  malih agenata koji odlučuju šta im treba pre nego što formiraju odgovor.
""".strip()


def _demo() -> None:
    """Mali demo toka i ispis rezultata na stdout.

    Napomena: mrežni delovi (OpenAI i Google) zahtevaju ključeve i Internet.
    """

    prompt = (
        "Hi! I’d like to cook a good Italian dish for lunch! "
        "I have potatoes, carrots, rosemary, and pork. Can you recommend a recipe and tips?"
    )
    try:
        answer = run_agent(prompt)
        print("\n=== FINAL ANSWER ===\n")
        print(answer)
    except Exception as e:  # edukativno: sandbox okruženja često blokiraju mrežu
        print(f"[warn] Demo nije uspeo ili je delimično preskočen: {e}")

    print("\nLEARNING NOTES:\n" + LEARNING_NOTES)


if __name__ == "__main__":
    _demo()

