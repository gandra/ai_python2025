# Clean/Remove Environment

## Conda

## Cišćenje starog conda okruženja

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
