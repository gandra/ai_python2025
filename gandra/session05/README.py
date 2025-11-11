"""
Programmable Search Engine (PSE) — Setup i Konfiguracija

Ovaj dokument te vodi korak‑po‑korak kroz podešavanje Google Programmable Search Engine (PSE)
i Custom Search JSON API ključeva koje koristi `gandra/session05/session05.py`.

1) Kreiraj PSE (Search engine ID — cx)
- Otvori: https://programmablesearchengine.google.com/
- Kreiraj novi Search Engine. Ako želiš opštu web pretragu, izaberi opciju da pretražuje ceo web.
- Sačuvaj „Search engine ID (cx)“ — koristićeš ga kao vrednost promenljive okruženja GOOGLE_CSE_CX.

2) Uključi Custom Search JSON API i napravi API ključ
- Otvori Google Cloud Console: https://console.cloud.google.com/
- Odaberi projekat (ili kreiraj novi), zatim „APIs & Services“ → „Library“.
- Pronađi „Custom Search API“ i klikni „Enable“.
- U „APIs & Services“ → „Credentials“ klikni „Create credentials“ → „API key“. Sačuvaj vrednost za GOOGLE_CSE_API_KEY.

Razlika: GOOGLE_CSE_CX vs GOOGLE_CSE_API_KEY
- GOOGLE_CSE_CX (Search engine ID — „cx“): identifikator tvog PSE indeksa. Određuje šta i kako pretražuješ (ceo web ili ograničene sajtove, SafeSearch, jezik…). Menjanjem PSE podešavanja menjaš rezultate bez promene koda. Nije tajna po sebi, ali je konfiguracija — drži ga u env‑u radi konzistentnosti.
- GOOGLE_CSE_API_KEY: API ključ iz Google Cloud‑a za autorizaciju poziva ka Custom Search JSON API‑ju. Veže se za projekat i kvote. Ovo je tajna — ne komituj; ograniči ključ na „Custom Search API“ i po potrebi na referrers/IP.
- Oba su potrebna: bez `cx` dobijaš grešku tipa „Missing cx“/„Invalid cx“, a bez `key` 403/unauthorized.

3) Kvote, bezbednost, podešavanja
- Custom Search API ima kvote i ograničenja — proveri „Quotas“ u Cloud Console‑u.
- PSE podešavanja (SafeSearch, jezik, zemlje) možeš podešavati u PSE konzoli (npr. SafeSearch=active).
- Ako želiš pretragu ograničenu na određene sajtove, u PSE dodeli listu domena.

4) Lokalna konfiguracija (env)
- Postavi promenljive okruženja u shell‑u ili `.env` fajlu (projekt ima `python-dotenv`).

 export OPENAI_API_KEY='sk-...'
  export GOOGLE_CSE_API_KEY='AIza...'
  export GOOGLE_CSE_CX='xxxxxxxxxxxxxxxxx'

- Alternativno `.env` (u root‑u projekta):

  OPENAI_API_KEY=sk-...
  GOOGLE_CSE_API_KEY=AIza...
  GOOGLE_CSE_CX=xxxxxxxxxxxxxxxxx

5) Brzi test (Python)

  import os, requests
  url = 'https://www.googleapis.com/customsearch/v1'
  params = {'key': os.getenv('GOOGLE_CSE_API_KEY'), 'cx': os.getenv('GOOGLE_CSE_CX'), 'q': 'ragù alla napoletana', 'num': 3}
  print(requests.get(url, params=params, timeout=20).json())

6) Pokretanje demo agenta
- `uv sync`
- `uv run python gandra/session05/session05.py`

Napomene
- Ne komituj tajne (ključeve) — `.env` je već ignorisan u .gitignore.
- Ako želiš fiksni jezik, postavi `GOOGLE_LR=lang_en` (ili ne postavljaj da ostane podrazumevano).
"""
