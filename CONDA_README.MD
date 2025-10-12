# Python for Data Science and Machine Learning Bootcamp

Repo for Python Data Science and Machine Learning Bootcamp

## Conda Environment & Setup general

```
source /opt/anaconda3/bin/activate

# (opciono) da se base ne pali automatski kad otvoriš shell
conda config --set auto_activate_base false

# kreiraj env sa Python 3.12 (u folderu projekta)
conda create -p ./env -c conda-forge -y python=3.12

# aktiviraj
conda activate ./env

# deaktiviraj
conda deactivate

# proveri
python --version

```

## Jupyter

```
# Покретање Jupyter-а
# Најчешће ћеш у прегледачу видети садржај кроз Jupyter Notebook или JupyterLab.
jupyter notebook
jupyter lab

# Ако желиш web интерфејс за управљање свим окружењима и пакетима, Anaconda има и Anaconda Navigator.
anaconda-navigator
```

## Env setup (Python 3.12, sve iz conda-forge gde ima smisla)
```
# 0) (opciono) ne aktiviraj base automatski
conda config --set auto_activate_base false

# 1) kreiraj env sa Python 3.12 (u folderu projekta)
conda create -p ./env -c conda-forge -y python=3.12

# 2) aktiviraj
conda activate ./env

# 3) core naučni/nativni paketi (conda-forge)
conda install -y -c conda-forge \
  numpy pandas scipy scikit-learn seaborn \
  jupyterlab ipykernel \
  spacy nltk \
  libjpeg-turbo libpng zlib


# 4) OpenAI i Transformers (pip u aktiviranom conda env-u)
python -m pip install --upgrade pip
pip install --upgrade openai transformers

```

### Preporuke ukratko

- **Ne mešaj kanale**: izaberi conda-forge i drži se njega za native/numeričke libove (numpy, pandas, sklearn, seaborn, spacy, jupyter…), baš kako si i zabeležio.

- **Teški/nativni paketi** → conda-forge, jer donosi kompatibilne binarne build-ove (i čak dodatne system libove poput libjpeg-turbo, libpng, zlib).

- **Čisto Python SDK-ovi (“openai”, često i “transformers”)** → pip u aktiviranom conda env-u, da dobiješ najnovije verzije bez zavisnosti od conda kanala.

- Ako ti treba **PyTorch/torchvision**, instaliraj ih preko pytorch + conda-forge kanala (ne pip), zbog nativnih zavisnosti. 

### spaCy modeli (preko spaCy CLI)

```
python -m spacy download en_core_web_sm
# po potrebi: en_core_web_md ili en_core_web_lg

```

### NLTK resursi (po potrebi)

```
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('wordnet')
PY

```

## Jupyter kernel (da se vidi u menijima)
```
python -m ipykernel install --user --name py312-env --display-name "Python 3.12 (project env)"

```


