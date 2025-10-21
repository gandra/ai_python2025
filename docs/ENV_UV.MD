# UV

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
