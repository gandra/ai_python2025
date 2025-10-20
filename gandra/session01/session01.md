# Session 01 — Csak kroz notebook `session01.ipynb`

## 1. Kako čitati notebook
Notebook je niz ćelija. Markdown ćelije nose objašnjenja, a code ćelije sadrže Python kod koji pokrećeš redom od vrha ka dnu. Svaka sledeća ćelija često zavisi od rezultata prethodne, pa je redosled važan.

## 2. Cell 1 — Naslov časa
- Sadržaj: `# GenAI/RAG in Python 2025` i `## Session 01. A Basic RAG Framework`.
- Značenje: ime kursa i tema lekcije. RAG (Retrieval-Augmented Generation) kombinuje pretragu teksta i generisanje odgovora.
- Akcija: nema, ovo je samo informacija.
- Summary: Kao naslov udžbeničkog poglavlja — najavljuje da ćemo praviti mini sistem koji nalazi recepte i piše odgovor umesto nas.

## 3. Cell 2 — Uvoz biblioteka
- Kod:
  ```python
  import os
  import numpy as np
  import pandas as pd
  from openai import OpenAI
  ```
- Objašnjenje:
  - `os` — most prema operativnom sistemu (npr. čitanje API ključeva).
  - `numpy (np)` — alat za rad s velikim nizovima brojeva (osnova ML matematike).
  - `pandas (pd)` — kao Excel u Pythonu: tabele, filtriranje, obrada.
  - `OpenAI` — klijent biblioteke za pozivanje OpenAI API-ja.
- Šta uraditi: ako ove biblioteke nisu instalirane, u terminalu pokreni `pip install numpy pandas openai`.
- Summary: Pripremamo alatku kutiju — matematiku (`numpy`), rad s tabelama (`pandas`) i pristup OpenAI servisima (`OpenAI`).

## 4. Cell 3 — Uvod u podatke
- Tekst najavljuje da će primer podataka biti italijanski recepti.
- Summary: Pripremi se, radićemo s receptima iz Italije.

## 5. Cell 4 — Učitavanje CSV fajla
- Kod:
  ```python
  file_path = "_data/italian_recipes_clean.csv"
  df = pd.read_csv(file_path)
  print(df.info())
  print(df.head())
  ```
- Objašnjenje:
  - `pd.read_csv` čita CSV u `DataFrame` `df`.
  - `df.info()` pokazuje strukturu (220 redova, kolone `title` i `receipt`).
  - `df.head()` prikazuje prvih 5 recepata.
- Šta uraditi: pokreni ćeliju; ako izbaci `FileNotFoundError`, proveri da fajl postoji u `_data`.
- Summary: U memoriji imamo tabelu od 220 recepata, svaki sa naslovom i tekstom.

## 6. Cell 5 — Opis sistema koji gradimo
- Tekst objašnjava 4 koraka: pitanje korisnika → pretraga po receptima → top 5 → ChatGPT sastavlja odgovor.
- Summary: Kao da pitaš kuvara šta da skuhaš s krompirom i šargarepom — on nađe recepte koji se uklapaju i lepo ti prepriča kako da ih spremiš.

## 7. Cell 6 — Šta su embeddings
- Embedding pretvara tekst u dugačku listu brojeva (vektor) koji predstavlja značenje.
- Moguće poređenje: svaki recept postaje tačka u ogromnoj (1536-dimenzionoj) mapi značenja. Ako se dva recepta bave sličnim stvarima, njihove tačke leže blizu.
- Model naučen na ogromnom korpusu tekstova uči da reči i rečenice sa sličnim značenjem projekuje na slične vektore.
- Summary: Embedding je numerički otisak prsta teksta — omogućava računaru da meri koliko su dva recepta tematski bliska.

## 8. Cell 7 — Pravimo embeddinge za recepte
- Kod pravi embedding za svaki recept, korak po korak:
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
      resp = client.embeddings.create(model=model_name, input=[text])
      emb = resp.data[0].embedding
      embeddings.append(emb)
  df["embedding"] = embeddings
  df.head()
  ```
- Objašnjenje:
  - `OPENAI_API_KEY` mora biti postavljen u okruženju (macOS/Linux `export OPENAI_API_KEY="tvoj_kljuc"`; Windows PowerShell `setx OPENAI_API_KEY "tvoj_kljuc"` pa novi prozor).
  - `client = OpenAI(...)` kreira klijenta koji zna da pozove API.
  - Petlja šalje tekst svakog recepta u OpenAI i dobija vektor od 1536 brojeva.
  - Rezultat čuvamo u novoj koloni `embedding`.
- Šta uraditi: pre pokretanja obezbedi API ključ i računaj da ova petlja šalje ~220 zahteva (troši vreme i kredite). Ako želiš, posle izvršavanja sačuvaj rezultate (`df.to_pickle(...)`) da drugi put ne plaćaš ponovo.
- Summary: Svakom receptu dodelili smo njegov numerički otisak — sada možemo matematički da ih upoređujemo.

## 9. Cell 8 — Provera tipa embeddinga
- Kod: `type(df["embedding"][0])` → rezultat `<class 'list'>`.
- Summary: Potvrdili smo da su embeddingi liste brojeva, kako i očekujemo.

## 10. Cell 9 — Dimenzija embeddinga
- Kod: `len(df["embedding"][0])` → 1536.
- Summary: Svaki recept je predstavljen vektorom od 1536 komponenti — to je rezolucija modela `text-embedding-3-small`.

## 11. Cell 10 — Najava korisničkog upita
- Markdown uvod: vreme je da zapišemo šta korisnik pita.
- Summary: Pripremamo “glas korisnika” koji će sistem analizirati.

## 12. Cell 11 — Tekst pitanja korisnika
- Kod:
  ```python
  user_text = """
  Hi! I’d like to cook ...
  """
  ```
- Objašnjenje:
  - Višelinijski string opisuje scenario: korisnik ima krompir, šargarepu, ruzmarin i svinjetinu i želi savete.
  - Po želji promeni tekst i pokreni ćeliju.
- Summary: U notebook smo upisali konkretno pitanje koje će RAG sistem obrađivati.

## 13. Cell 12 — Podsetnik na embedding pitanja
- Markdown: i korisničko pitanje moramo pretvoriti u embedding da bismo ga poredili s receptima.
- Summary: Pitanje će dobiti svoj numerički otisak, isto kao recepti.

## 14. Cell 13 — Embedding korisničkog pitanja
- Kod poziva OpenAI embedding API za `user_text`.
- `user_query` postaje lista od 1536 brojeva.
- `print` potvrđuje tip i dužinu.
- Šta uraditi: pokreni posle što je `client` definisan. Ako dobiješ grešku, proveri API ključ.
- Summary: Sad imamo numeričku reprezentaciju i za korisnikov upit.

## 15. Cell 14 — Najava pretrage
- Markdown uvodi sledeću fazu: pronalazak najprikladnijih recepata.
- Summary: Sledeći korak je matematičko poređenje između pitanja i svakog recepta.

## 16. Cell 15 — Računanje sličnosti i top 5 recepata
- Kod koristi kosinusnu sličnost da oceni svaki recept, zatim bira pet najboljih i spaja ih u `prompt_recipes`.
- Bitni momenti:
  - `from scipy.spatial.distance import cosine` za kosinusnu distancu.
  - `scores.append(1.0 - cosine(...))` jer manja distanca znači veća sličnost.
  - `np.argsort(scores)[-5:]` uzima indekse 5 najvećih skorova.
  - U petlji gradimo tekst koji sadrži naslov i ceo recept.
- Šta uraditi: instaliraj SciPy (`pip install scipy`) ako nije tu. Po želji ispiši vrednosti skorova radi uvida.
- Summary: Izmerili smo koliko svaki recept liči na korisničko pitanje i uzeli pet najboljih — ti recepti će poslužiti kao “materijal” za finalni odgovor.

## 17. Cell 16 — Matematika kosinusne sličnosti
- Formule:
  - `cosθ = (a · b) / (||a|| ||b||)` — ugao između vektora.
  - `d_cos = 1 - cosθ` — kosinusna distanca.
- Uloga: veći `cosθ` ⇒ manji ugao ⇒ sličniji tekstovi.
- Summary: Kosinusna sličnost meri koliko dva vektora “gledaju” u istom pravcu; to je baza poređenja embeddinga.

## 18. Cell 17 — Ulaz u fazu generisanja
- Markdown: “Finally, use an LLM to shape the final response.”
- Summary: Sada ćemo sakupljene recepte i pitanje proslediti LLM-u da napiše završni odgovor.

## 19. Cell 18 — Sastavljanje prompta za LLM
- F-string prompt uključuje:
  - Ulogu asistenta (“You are a helpful Italian cooking assistant.”).
  - Niz recepata (`prompt_recipes`).
  - Korisničko pitanje (`user_text`).
  - Jasna pravila (prepoznaj relevantno, ignoriši nebitno, prepričaj prijateljski, koristi naslove, predloži više jela).
- Zašto je važno: što preciznije objasnimo šta želimo, to model daje korisniji odgovor.
- Šta uraditi: pokreni ćeliju — ona samo formira tekst, ne šalje API zahtev. Izmeni uputstva ako želiš drugačiji stil odgovora.
- Summary: Spremili smo kompletan paket informacija i instrukcija koje ćemo poslati ChatGPT-u.

## 20. Cell 19 — Poziv ChatGPT-u i tumačenje role
- Kod:
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
- Uloge:
  - `system` — “učitelj” koji modelu kaže kako da se ponaša (ovde: glumi ljubaznog italijanskog kuvara). Ovo ima najveću težinu.
  - `user` — šta korisnik (mi) traži; ubacujemo ceo prompt sa receptima i pravilima.
  - `assistant` — (nema ga u ovom primeru) bi sadržavao ranije odgovore modela.
- Parametri:
  - `model` — izabrani ChatGPT model (ako nemaš pristup `gpt-4`, koristi `gpt-4o-mini` ili `gpt-3.5-turbo`).
  - `temperature` — kontrola kreativnosti (manje vrednosti daju strožije, predvidljivije odgovore).
  - `max_tokens` — maksimalna dužina odgovora.
- Rezultat: iz `response` vadimo tekst (`reply_text`) i prikazujemo ga.
- Šta uraditi: pokreni ćeliju, proveri da li imaš pristup modelu i API ključ. Ovaj korak troši kredite.
- Summary: Poslali smo ulogu i informacije modelu, a zatim dobili finalnu priču koja predlaže jela i savete, prilagođenu onome šta korisnik ima.

## 21. Šta treba uraditi da notebook radi
1. Instaliraj biblioteke: `pip install numpy pandas scipy openai`.
2. Postavi promenljivu okruženja `OPENAI_API_KEY` s tvojim OpenAI ključem.
3. Otvori notebook i izvršavaj ćelije redom od prve do poslednje.
4. Po želji sačuvaj DataFrame s embeddingom (`df.to_pickle(...)`) da sledeći put preskočiš API pozive u Cell 7.

## 22. Ideje za dalje eksperimente
1. Menjaj `user_text` i posmatraj kako se menjaju top recepti i odgovor.
2. U Cell 15 ispiši vrednosti `scores` da vidiš numeričku sličnost svakog recepta.
3. Dodaj sortiranje top rezultata po opadajućem scores (da prvi bude najbolji).
4. Sačuvaj embeddinge i učitavaj ih umesto ponovnog pozivanja API-ja.
5. Promeni `temperature` ili instrukcije u promptu i prouči kako to utiče na stil odgovora.

