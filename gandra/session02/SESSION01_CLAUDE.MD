# RAG Framework u Python-u - Kompletno Objašnjenje za Početnike

## Šta je RAG (Retrieval-Augmented Generation)?

RAG je tehnika u veštačkoj inteligenciji koja kombinuje **pretraživanje informacija** (Retrieval) sa **generisanjem teksta** (Generation). Umesto da AI model generiše odgovore samo na osnovu svog ugrađenog znanja, RAG prvo pronađe relevantne informacije iz baze podataka, a zatim koristi te informacije za generisanje boljeg odgovora.

### Zašto je RAG koristan?
- **Tačnost**: Smanjuje halucincije AI modela jer se oslanja na konkretne podatke
- **Aktuelnost**: Možete koristiti najnovije podatke bez ponovnog treniranja modela
- **Kontrola**: Imate potpunu kontrolu nad izvorom informacija

---

## 1. Uvoz Potrebnih Biblioteka

```python
import os
import numpy as np
import pandas as pd
from openai import OpenAI
```

### Objašnjenje biblioteka:
- **`os`**: Omogućava rad sa operativnim sistemom (čitanje promenljivih okruženja za API ključ)
- **`numpy`**: Biblioteka za numeričke operacije, posebno rad sa vektorima
- **`pandas`**: Biblioteka za rad sa tabelarnim podacima (CSV fajlovi, DataFrame strukture)
- **`OpenAI`**: Python klijent za pristup OpenAI API servisu (ChatGPT, embeddings)

---

## 2. Učitavanje i Priprema Podataka

```python
file_path = "_data/italian_recipes_clean.csv"
df = pd.read_csv(file_path)
```

### Šta se dešava:
1. **Učitava se CSV fajl** sa italijanskim receptima
2. **DataFrame struktura**: Pandas organizuje podatke u tabelu sa kolonama `title` (naziv jela) i `receipt` (recept)
3. Imamo **220 recepata** spremnih za pretraživanje

### Struktura podataka:
- Svaki red predstavlja jedan recept
- Kolona `title`: naziv jela (npr. "BREAD SOUP")
- Kolona `receipt`: detaljan opis kako se priprema

---

## 3. Vector Embeddings - Srce Sistema

### Šta su Vector Embeddings?

**Embeddings** su način da se tekst pretvori u numeričke vektore koje računar može da razume i poredi. Svaki tekst se pretvara u niz brojeva (vektor) koji predstavlja semantičko značenje tog teksta.

```python
# Povezivanje sa OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Izbor modela za embeddings
model_name = "text-embedding-3-small"
```

### Proces kreiranja embeddings-a:

```python
embeddings = []
for idx, row in df.iterrows():
    text = row["receipt"]
    
    # Poziv OpenAI API-ja za kreiranje embedding-a
    resp = client.embeddings.create(
        model=model_name,
        input=[text]
    )
    
    # Čuvanje embedding vektora
    emb = resp.data[0].embedding
    embeddings.append(emb)

df["embedding"] = embeddings
```

### Ključni koncepti:
- **Svaki recept** se pretvara u vektor od **1536 brojeva**
- **Slični tekstovi** imaju slične vektore u multidimenzionalnom prostoru
- **Model `text-embedding-3-small`**: OpenAI-jev model optimizovan za brzinu i kvalitet

---

## 4. Obrada Korisničkog Upita

```python
user_text = """
Hi! I'd like to cook a good Italian dish for lunch! I have potatoes, carrots, 
rosemary, and pork. Can you recommend a recipe and help me a bit with 
preparation tips?
"""

# Kreiranje embedding-a za korisnički upit
resp = client.embeddings.create(
    model=model_name,
    input=[user_text]
)
user_query = resp.data[0].embedding
```

### Šta se dešava:
1. **Isti proces** kao za recepte - tekst se pretvara u vektor
2. **Cilj**: omogućiti poređenje korisničkog upita sa svim receptima
3. **Rezultat**: vektor od 1536 brojeva koji predstavlja korisnički zahtev

---

## 5. Similarity Search - Pronalaženje Najboljih Recepata

### Cosine Similarity

**Cosine similarity** meri ugao između dva vektora. Što je ugao manji, vektori su sličniji.

```python
from scipy.spatial.distance import cosine

scores = []
for emb in df["embedding"]:
    if emb is None:
        scores.append(-1.0)
    else:
        # Similarity = 1 - cosine_distance
        scores.append(1.0 - cosine(np.array(emb), np.array(user_query)))
```

### Matematička formula:
$$\text{similarity} = 1 - \text{cosine\_distance} = \cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}$$

Gde su:
- **a** i **b**: dva vektora koje poredimo
- **a · b**: skalarni proizvod vektora
- **|a|** i **|b|**: dužine vektora

### Izbor top 5 recepata:

```python
# Sortiranje i uzimanje 5 najboljih
top5 = np.argsort(scores)[-5:]

# Kreiranje stringa sa najboljim receptima
output_lines = []
for i in top5:
    title = df.iloc[i]["title"]
    recipe = df.iloc[i]["receipt"]
    output_lines.append(f"{title}:\n{recipe}")
prompt_recipes = "\n\n".join(output_lines)
```

### Objašnjenje `np.argsort()`:
- **`np.argsort(scores)`**: vraća indekse koji bi sortirali niz u rastućem redosledu
- **`[-5:]`**: uzima poslednjih 5 indeksa (oni sa najvećim skorovima)

---

## 6. Generisanje Finalnog Odgovora sa LLM

### Kreiranje Prompt-a

```python
prompt = f"""
You are a helpful Italian cooking assistant.  
Here are some recipe examples I found that may or may not be relevant to the user's request:

{prompt_recipes}

User's question: "{user_text}"

From the examples above:
1. Determine which recipes are *relevant* to what the user asked and which are not.
2. Discard or ignore irrelevant ones, and focus on relevant ones.
3. For each relevant example, rephrase the recipe in a more narrative, 
conversational style...
"""
```

### Ključni elementi prompt engineering-a:
1. **Definisanje uloge**: "You are a helpful Italian cooking assistant"
2. **Kontekst**: Davanje pronađenih recepata
3. **Jasne instrukcije**: Korak po korak šta AI treba da uradi
4. **Format izlaza**: Kako treba da izgleda finalni odgovor

### Poziv ChatGPT API-ja:

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
```

### Parametri:
- **`model="gpt-4"`**: Koristi se GPT-4 model
- **`temperature=1`**: Kontroliše kreativnost (0=deterministički, 2=vrlo kreativno)
- **`max_tokens=5000`**: Maksimalna dužina odgovora
- **`messages`**: Lista poruka sa ulogama (system/user/assistant)

---

## Kako Ceo Sistem Funkcioniše - Korak po Korak

### 1. **Priprema Phase**
   - Učitava se baza recepata
   - Svaki recept se pretvara u vektor (embedding)
   - Čuva se u DataFrame strukturi

### 2. **Query Phase**
   - Korisnik postavlja pitanje
   - Pitanje se pretvara u vektor

### 3. **Retrieval Phase**
   - Poredi se vektor pitanja sa svim vektorima recepata
   - Računa se cosine similarity za svaki par
   - Bira se top 5 najsličnijih recepata

### 4. **Generation Phase**
   - Top 5 recepata se pakuje u prompt
   - GPT-4 analizira relevantnost
   - GPT-4 generiše personalizovan odgovor

### 5. **Output**
   - Korisnik dobija detaljan, narativni odgovor
   - Odgovor sadrži konkretne recepte prilagođene sastojcima

---

## Prednosti RAG Sistema

### 1. **Tačnost**
- Sistem se oslanja na konkretne recepte, ne izmišlja

### 2. **Relevantnost**
- Similarity search pronalazi najrelevantnijih recepte

### 3. **Personalizacija**
- GPT-4 prilagođava odgovor specifičnom upitu

### 4. **Skalabilnost**
- Može raditi sa hiljadama recepata

---

## Moguća Poboljšanja

### 1. **Hybrid Search**
- Kombinovanje semantic search sa keyword search

### 2. **Metadata Filtering**
- Filtriranje po kategorijama (predjelo, glavno jelo, desert)

### 3. **Re-ranking**
- Dodatno rangiranje rezultata na osnovu više faktora

### 4. **Cache System**
- Čuvanje embedding-a da se izbegnu ponovni API pozivi

### 5. **Vector Database**
- Korišćenje specijalizovanih baza (Pinecone, Weaviate, Chroma)

---

## Zaključak

Ovaj kod demonstrira osnovni RAG pipeline koji:
1. **Transformiše tekst u vektore** za semantičko pretraživanje
2. **Pronalazi relevantne informacije** kroz similarity search
3. **Generiše kvalitetne odgovore** kombinovanjem pronađenih podataka sa LLM-om

RAG je moćna tehnika koja omogućava AI sistemima da budu tačniji, pouzdaniji i korisniji u realnim aplikacijama. Ovaj primer sa receptima može se lako prilagoditi za druge domene kao što su:
- Tehnička dokumentacija
- Customer support
- Pravni dokumenti
- Medicinske informacije
- Obrazovni sadržaj

---

## Dodatni Resursi za Učenje

- **OpenAI Documentation**: https://platform.openai.com/docs
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **NumPy Documentation**: https://numpy.org/doc/
- **Vector Databases**: Pinecone, Weaviate, Chroma, Qdrant
- **LangChain Framework**: Za naprednije RAG implementacije