# PostgreSQL + pgvector preko Docker Compose

Ovaj dokument objašnjava kako da pokreneš lokalnu PostgreSQL instancu sa [pgvector](https://github.com/pgvector/pgvector) ekstenzijom koristeći `docker-compose.yml` fajl u root-u repozitorijuma.

## 1. Preduslovi
- Instaliran Docker Engine / Docker Desktop (verzija 20.10+).
- Docker Compose (`docker compose` CLI je preporučen; `docker-compose` i dalje radi ako ga koristiš).
- Otvoren port `5432` na mašini ili izaberi drugi port u `docker-compose.yml`.

## 2. Konfigurisanje instanci
`docker-compose.yml` kreira servis `pgvector` sa podrazumevanim kredencijalima:

```yaml
POSTGRES_DB=ai_rag
POSTGRES_USER=ai_rag
POSTGRES_PASSWORD=ai_rag_password
```

Po potrebi ažuriraj vrednosti direktno u `docker-compose.yml` ili dodaj `.env` fajl pored njega i prebaci promenljive (Compose će ih automatski povući).

Podaci se čuvaju u named-volumu `pgvector-data`, pa restart kontejnera ne briše bazu.

## 3. Pokretanje
U root folderu repozitorijuma pokreni:

```bash
docker compose up -d pgvector
```

Proveri da li je servis spreman:

```bash
docker compose ps
docker compose logs -f pgvector
```

Zdravstveniček (`healthcheck`) proverava dostupnost baze, pa status treba da pređe u `healthy` nakon inicijalizacije.

## 4. Prvi pristup i pgvector ekstenzija

### Ručno kreiranje
Poveži se preko `psql` i aktiviraj ekstenziju (u većini slučajeva dovoljno je uraditi jednom):

```bash
docker compose exec -it pgvector psql -U ai_rag -d ai_rag
ai_rag=# CREATE EXTENSION IF NOT EXISTS vector;
ai_rag=# \dx
ai_rag=# \q
```

Ako si promenio/la kredencijale, zameni `ai_rag` vrednostima koje koristiš.

### Automatsko kreiranje pri prvom pokretanju
PostgreSQL inicijalizacioni skriptovi iz direktorijuma `/docker-entrypoint-initdb.d` se izvršavaju samo kada se baza pokreće prvi put (tj. kada je data direktorijum prazan). To možemo iskoristiti da se `vector` ekstenzija kreira automatski:

1. Kreiraj lokalni direktorijum, npr. `docker/initdb` i u njemu fajl `01-enable-pgvector.sql` sa sadržajem:

    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```

2. U `docker-compose.yml` dodaj mount za taj direktorijum (ili ga ubaci u postojeću `volumes` sekciju):

    ```yaml
    services:
      pgvector:
        volumes:
          - ./docker/initdb:/docker-entrypoint-initdb.d
    ```

3. Obriši postojeći named-volume samo ako želiš da testiraš inicijalizaciju iz početka (`docker compose down -v`). Kod prvog sledećeg `docker compose up -d pgvector` Postgres će pokrenuti skript i ekstenzija će biti kreirana pre nego što se bilo koja aplikacija poveže na bazu.

Ovaj pristup ne utiče na restart postojećeg kontejnera – ekstenzija će ostati aktivna jer je deo same baze, dok se skript izvršava samo pri prvom kreiranju volumena.

## 5. Zaustavljanje i čišćenje
- Zaustavljanje: `docker compose down`
- Zaustavljanje + brisanje volumena sa podacima: `docker compose down -v`

## 6. Platforma-specifični saveti

### macOS
- Instliraj [Docker Desktop za macOS](https://www.docker.com/products/docker-desktop/).
- Ako koristiš Apple Silicon, Docker automatski povlači odgovarajuću ARM sliku; nema dodatnih koraka.
- Komande iz sekcija iznad pokreći iz Terminal aplikacije (`zsh` ili `bash`).

### Linux
- Uveri se da korisnik ima prava za Docker (članstvo u `docker` grupi ili `sudo`).
- Ako servis ne startuje posle restarta, omogući `docker` servis (`sudo systemctl enable --now docker`).
- Sve komande iz sekcija iznad rade identično; dodaj `sudo` ako Docker zahteva administratorske privilegije.

### Windows
- Instaliraj [Docker Desktop za Windows](https://www.docker.com/products/docker-desktop/) i aktiviraj WSL 2 backend.
- Otvori PowerShell ili Windows Terminal i pokreni komande iz sekcija iznad. Alternativno, koristi WSL distribuciju (Ubuntu, Debian) i radi iz Linux okruženja.
- Ako koristiš klasični `docker-compose`, zameni `docker compose` pozive sa `docker-compose`.

## 7. Sledeći koraci
- Poveži aplikaciju koristeći connection string `postgresql://ai_rag:ai_rag_password@localhost:5432/ai_rag` (prilagodi vrednosti prema konfiguraciji).
- Kreiraj tabelu sa `vector` kolonama i testiraj upite, npr. `VECTOR(1536)` za OpenAI embeddinge.
