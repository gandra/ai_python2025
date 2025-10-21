# UV

<a id="cleanup-conda"></a>
## Čišćenje starog `conda` okruženja

1. Uveri se da nijedan `conda` environment nije aktivan:
   ```bash
   conda deactivate
   ```
2. Ako je projekat koristio lokalni environment u `./env`, ugasi ga i obriši:
   ```bash
   # opciono proveri gde je napravljen env
   conda env list | grep ai_python2025

   # ukloni environment kreiran sa -p ./env
   conda env remove -p "$(pwd)/env"
   # ako iz nekog razloga remove ne prođe, obriši ručno
   rm -rf env
   ```
3. (Opcionalno) Osveži shell konfiguraciju uklanjanjem referenci na `conda` ako ti više nisu potrebne (`~/.zshrc`, `~/.bashrc`).

<a id="cleanup-uv"></a>
## Brisanje postojećeg `uv` okruženja

1. Ako je `.venv` trenutno aktivan, izađi iz njega:
   ```bash
   deactivate  # preskoči ako nisi aktivirao/aktivirala ručno
   ```
2. Obriši virtuelno okruženje koje `uv` koristi u projektu:
   ```bash
   rm -rf .venv
   ```
   Ako koristiš prilagođenu putanju (`uv venv --path path/do/okruzenja`), obriši odgovarajući folder.
3. Ako si ranije postavljao/la lokalni `UV_CACHE_DIR` (npr. `.uv-cache` u projektu), možeš da ga očistiš:
   ```bash
   rm -rf .uv-cache
   ```
   Globalni cache možeš obrisati komandom `uv cache clean`.

## Instalacija `uv`

1. Instaliraj `uv` (single-binary instalacija):
   ```bash
   curl -Ls https://astral.sh/install.sh | sh
   ```
2. Script će smestiti binarije u `~/.local/bin` (ili sličan direktorijum). Dodaj ga u `PATH` ako već nije:
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```
3. Proveri instalaciju:
   ```bash
   uv --version
   ```

> Ako želiš da `uv` upravlja i Python binarijama, možeš instalirati željenu verziju:
> ```bash
> uv python install 3.12
> ```
> Kasnije prilikom kreiranja okruženja možeš eksplicitno ciljati verziju pomoću `--python 3.12`.
