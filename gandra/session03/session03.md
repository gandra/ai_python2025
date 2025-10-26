Dodao sam gandra/session03/session03.py koji replicira logiku iz beležnice i uvodi jasnu strukturu sa funkcijama, bogatim komentarima i sažecima za učenje. Skripta pokriva ingest u Postgres+pgvector, vektorske upite sličnosti, OpenAI embeddings i chat, kao i primer Function Calling-a za vreme.

Šta je urađeno
- Učitavanje i priprema podataka
  - load_recipes_with_embeddings: čita `_data/italian_recipes_embedded.csv`, parsira kolonu embedding (string) u numpy vektore float32, dodaje kolonu `embedding_vector`.
- Postgres + pgvector
  - connect_postgres: konektuje se na lokalni Docker Postgres i registruje pgvector adapter.
  - reset_and_create_table: drop/create tabele `receipts` sa `VECTOR(dim)`.
  - ingest_dataframe: upisuje sve redove (title, receipt, embedding).
- Embedding i RAG upit
  - build_openai_client: kreira OpenAI klijent ako postoji `OPENAI_API_KEY`.
  - embed_text: embedding korisničkog teksta.
  - query_similar_recipes: SQL sa `embedding <=> %s::vector` i sličnost `1 - distance`, vraća DataFrame.
  - build_cooking_prompt i chat_cooking_assistant: gradi prompt sa pronađenim receptima i poziva Chat Completions (narativan odgovor).
- Function Calling primer (vreme)
  - get_current_weather: poziva `wttr.in` da vrati sažetak vremena (temperatura + opis).
  - run_function_calling_weather_demo: registruje alat (function schema), model bira alat i argumente, izvrši Python funkciju i vrati finalan odgovor.
- Orkestracija
  - run_demo_pipeline: end-to-end tok (ingest → embedding upit → prompt + chat → weather tool demo).
  - LEARNING_NOTES: sažetak najbitnijih koncepata za učenje na kraju.
- Ugrađeni uputi (u komentarima na vrhu)
  - Kako pokrenuti Postgres+pgvector kroz Docker.
  - Kako instalirati sa uv-om.
  - Poređenje psycopg (v3) vs psycopg2 (v2) sa preporukama.

Gde da gledaš
- Kod: gandra/session03/session03.py:1
- Glavne funkcije koje ćeš najviše čitati/proučavati:
  - gandra/session03/session03.py:66 (load_recipes_with_embeddings)
  - gandra/session03/session03.py:91 (connect_postgres)
  - gandra/session03/session03.py:109 (reset_and_create_table)
  - gandra/session03/session03.py:130 (ingest_dataframe)
  - gandra/session03/session03.py:149 (build_openai_client)
  - gandra/session03/session03.py:164 (embed_text)
  - gandra/session03/session03.py:174 (query_similar_recipes)
  - gandra/session03/session03.py:195 (build_cooking_prompt)
  - gandra/session03/session03.py:219 (chat_cooking_assistant)
  - gandra/session03/session03.py:238 (get_current_weather)
  - gandra/session03/session03.py:255 (run_function_calling_weather_demo)
  - gandra/session03/session03.py:305 (run_demo_pipeline)
  - gandra/session03/session03.py:350 (_print_notes i __main__)

psycopg vs psycopg2 (detaljno objašnjenje u komentarima)
- psycopg (v3, „psycopg“ paket):
  - Novi API, aktivno održavan, bolja asinhrona podrška, jednostavniji adapteri (npr. pgvector).
  - Preporučen za nove projekte.
- psycopg2 (v2, „psycopg2“/„psycopg2-binary“):
  - Stariji, veoma stabilan i rasprostranjen; gomila primera na internetu.
  - Manje moderan API i asink mogućnosti.
- Preporuka:
  - Novi projekti: psycopg (v3).
  - Legacy koji već radi na psycopg2: ostati na v2 dok ne postoji razlog za migraciju.
- Instalacija sa uv:
  - psycopg v3: uv add "psycopg[binary]"
  - psycopg2: uv add psycopg2-binary

Kako pokrenuti (lokalno)
- Podigni Postgres+pgvector Docker (komande su u zaglavlju skripte).
- Postavi OPENAI_API_KEY u okruženju.
- Pokreni: python gandra/session03/session03.py
  - Ako nemaš ključ ili mreža nije dostupna, skripta će ingestovati podatke i preskočiti OpenAI delove uz jasne poruke.
  - Na kraju ispisuje LEARNING_NOTES.

Sledeći koraci
- Da implementiram indeks na koloni embedding i pokazati razliku u performansama (pgvector HNSW/IVFFlat).
- Dodati CLI argumente (npr. --ingest-only, --query-only, --function-calling).
- Uvesti .env čitanje (python-dotenv) i zameniti hardkodovane vrednosti.
- Dodati unittest-e za transformacije (parsing embedding-a i formatiranje prompta).

Predlog git commit poruke
- Add session03 Python script: Postgres+pgvector ingest, OpenAI RAG, function calling demo with detailed comments and psycopg vs psycopg2 notes

