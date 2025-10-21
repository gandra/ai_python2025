# Prelazak sa Conda na uv package manager

Ovaj dokument opisuje kako da ukloniš postojeći `conda` environment, postaviš i održavaš novo Python okruženje koristeći [uv](https://docs.astral.sh/uv/). Uputstvo je pisano za macOS, ali je (uz manje prilagođavanje) primenljivo i na Linux.

## 0. Šta se menja
- Umesto `conda` koristi se `uv` za upravljanje Python verzijom, virtuelnim okruženjem, paketima i lock fajlovima.
- Okruženje se kreira lokalno u folderu `.venv` (po difoltu), a zavisnosti se zaključavaju kroz `uv.lock`.
- `requirements.txt` može ostati kao referenca, ali se preporučuje definisanje zavisnosti u `pyproject.toml` i održavanje preko `uv` komandi.

---

## 1. Priprema i čišćenje starog okruženja

[Priprema i čišćenje starog okruženja](ENV_CLEAN.MD)

---

## 2. Instalacija `uv`

[Instalacija uv](ENV_UV.MD)

---

## 3. Kreiranje i rad sa `uv` okruženjem u projektu

### 3.1. Inicijalizuj projekat
Ako projekat još nema `pyproject.toml`, sa root-a repozitorijuma pokreni:
```bash
cd /Users/draganmijatovic/learn/_COURSES/datakolektiv/src/ai_python2025_fork
uv init --bare
```
Komanda kreira minimalni `pyproject.toml` (bez sample koda) i `uv.lock` fajl. Ako `pyproject.toml` već postoji, preskoči ovaj korak i koristi postojeći fajl.

### 3.2. Kreiraj virtuelno okruženje
```bash
uv venv --python 3.12
```
Time dobijaš `.venv/` u root-u projekta. Nije potrebno ručno aktiviranje (`source`) osim ako to baš želiš.

> Ako radiš iz okruženja sa ograničenim pristupom fajl sistemu (npr. edukativni sandbox), postavi cache unutar projekta da bi komanda uspela:
> ```bash
> UV_CACHE_DIR=.uv-cache uv venv --python 3.12
> ```

### 3.3. Uvezi postojeće zavisnosti ili pređi na `pyproject.toml`
Postoje dva scenarija:

- **Brzi start (oslanja se na postojeći requirements):**
  ```bash
  uv pip install -r requirements.txt
  uv pip freeze > requirements.lock  # opciono snimi trenutno stanje
  ```
  Ovo populira `.venv` na osnovu trenutnog `requirements.txt`. Kasnije možeš preći na deklarativni pristup.

- **Potpuna migracija na `pyproject.toml`:**
  1. Otvori `pyproject.toml` i u okviru `[project]` sekcije dodaj/uredi:
     ```toml
     [project]
     name = "ai-python2025"
     version = "0.1.0"
     description = "Python data science sandbox"
     requires-python = ">=3.12,<3.13"
     dependencies = [
       "annotated-types==0.7.0",
       "anyio==4.11.0",
       "httpx==0.28.1",
       "ipykernel==6.30.1",
       "ipython==9.6.0",
       "jupyter-client==8.6.3",
       "jupyter-core==5.8.1",
       "numpy==2.3.3",
       "openai==2.1.0",
       "pandas==2.3.3",
       "scipy==1.16.2",
       "tqdm==4.67.1"
       # dodaj ostale pakete po potrebi
     ]
     ```
  2. Zavisnosti koje su samo za razvoj (npr. testiranje, lint) premesti u `[project.optional-dependencies.dev]` ili ih dodaj pomoću `uv add --dev`.
  3. Pokreni sinhronizaciju:
     ```bash
     uv sync
     ```
     Komanda kreira/obnavlja `.venv` i zaključava verzije u `uv.lock`.

> Tip: `uv add paket==verzija` i `uv remove paket` automatski ažuriraju `pyproject.toml` i `uv.lock`.

### 3.4. Rad sa okruženjem
`uv` po difoltu koristi `.venv` u root-u projekta, tako da nije potrebno ručno aktiviranje.

- Provera verzije:
  ```bash
  uv run python --version
  ```
- Pokretanje skripti / Jupytera:
  ```bash
  uv run python script.py
  uv run jupyter lab
  ```
- Dodavanje novih paketa:
  ```bash
  uv add seaborn
  uv add --dev ruff pytest
  ```
- Ako želiš klasično aktiviranje, pozovi:
  ```bash
  source .venv/bin/activate
  ```
  i kada završiš, isključi okruženje komandom:
  ```bash
  deactivate
  ```

### 3.5. Jupyter kernel registracija
Ako želiš da se uv okruženje pojavi u Jupyter menijima:
```bash
uv run python -m ipykernel install --user --name ai-python2025-uv --display-name "Python (uv)"
```

### 3.6. Instalacija biblioteka iz prethodnog conda setup-a
`CONDA_README.MD` navodi core pakete koje je conda instalirala. Isti set možeš da dobiješ preko `uv` sledećim koracima:

```bash
# osnovni naučni i Jupyter paketi
uv add numpy pandas scipy scikit-learn seaborn
uv add jupyterlab ipykernel

# NLP alati
uv add spacy nltk

# OpenAI SDK i Transformers
uv add openai transformers
```

> `uv add` automatski ažurira `pyproject.toml` i `uv.lock`. Ako koristiš "Brzi start" pristup sa `requirements.txt`, ove komande možeš preskočiti jer `uv pip install -r requirements.txt` već instalira iste pakete.

#### spaCy modeli
```bash
uv run python -m spacy download en_core_web_sm
# po potrebi: en_core_web_md ili en_core_web_lg
```

#### NLTK resursi
```bash
uv run python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('wordnet')
PY
```

#### Sistemske biblioteke (libjpeg-turbo, libpng, zlib)
`uv` ne instalira nativne biblioteke, pa ih obezbedi preko sistemskog paket menadžera. Na macOS-u (Homebrew):

```bash
brew install jpeg libpng zlib
```

Na Linux distribucijama potraži ekvivalentne pakete (npr. `sudo apt install libjpeg-turbo8-dev libpng-dev zlib1g-dev`).

> Ako ti treba PyTorch/torchvision, koristi zvanične `pip` wheel-ove u okviru `uv` okruženja (`uv add torch torchvision`), ali računaj na dodatne sistemske zavisnosti i veću instalaciju.


---

## 4. Resetovanje ili ponovno kreiranje okruženja
- Brisanje trenutnog virtuelnog okruženja:
  ```bash
  rm -rf .venv
  ```
- Ponovna sinhronizacija iz `pyproject.toml` / `uv.lock`:
  ```bash
  uv sync --reinstall
  ```
- Čišćenje keša (ako želiš da oslobodiš prostor):
  ```bash
  uv cache clean
  ```

---

## 5. Bitne napomene i ograničenja
- `uv` trenutno (Q1 2025) ne instalira sistemske biblioteke kao što to radi `conda`. Ako ti zatreba specifična C biblioteka, moraćeš da je instaliraš kroz `brew` ili sistemski paket menadžer.
- `uv` preferira rad sa `pyproject.toml`. Ako ostaneš na čistom `requirements.txt`, izgubićeš prednosti automatskog lock fajla i deklarativnog opisa projekta.
- Kada koristiš `uv run`, nije potrebno `source .venv/bin/activate`. Ako ipak želiš klasičan način rada:
  ```bash
  source .venv/bin/activate
  ```
- `uv sync` osvežava i `uv.lock`. Komituj i `pyproject.toml` i `uv.lock` kako bi tim mogao da reprodukuje okruženje.
- Nakon potpune migracije odaberi jednu „tačku istine“ za zavisnosti. Preporuka je da `requirements.txt` ili ukloniš ili ga generišeš iz `uv.lock` (npr. `uv export -o requirements.txt`).

---

## 6. Brzi spisak najčešćih komandi
```bash
# kreiraj / osveži .venv i instaliraj zavisnosti
uv sync

# dodaj paket u glavne zavisnosti
uv add pandas

# dodaj razvojnu zavisnost
uv add --dev pytest

# pokreni komandu u okviru okruženja
uv run jupyter lab

# ažuriraj sve zavisnosti na najnovije kompatibilne verzije
uv sync --upgrade

# izvezi requirements.txt ako ti je potreban za deljenje
uv export -o requirements.txt
```

---

Ako zapneš, `uv --help` i [zvanična dokumentacija](https://docs.astral.sh/uv/) imaju detaljne primere za sve korake.
