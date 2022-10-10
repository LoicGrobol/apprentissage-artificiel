---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

<!-- #region slideshow={"slide_type": "slide"} -->
Cours 4 : Pip et Virtualenv
===========================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-10-12
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Les modules sont vos meilleurs amis

Vraiment, promis

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Ils cachent vos implémentations

- Quand on code une interface, on a pas envie de voir le code fonctionnel
- Et vice-versa
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Ce qu'on fait quand on code proprement (ou presque)
<!-- #endregion -->

<!-- #region -->
```python
# malib.py
import re

def tokenize(s):
    return re.split(r"\s+", s)
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
```python
# moninterface.py
import sys
import malib

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as in_stream:
        for l in in_stream:
            print("_".join(list(malib.tokenize(l))))
```
<!-- #endregion -->

Comme ça quand je code mon interface, je n'ai pas besoin de me souvenir ou même de voir comment est codé `tokenize`

<!-- #region slideshow={"slide_type": "subslide"} -->
### Il y en a déjà plein
<!-- #endregion -->

```python
help("modules")
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Ils vous simplifient la vie
<!-- #endregion -->

```python
import pathlib

p = pathlib.Path(".").resolve()
display(p)
display(list(p.glob("*.ipynb")))
projet = p.parent / "assignments"
display(list(exos.glob("*")))
```

<!-- #region slideshow={"slide_type": "slide"} -->
## `stdlib` ne suffit plus

Parfois la bibliothèque standard est trop limitée
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Je veux repérer dans un texte toutes les occurrences de « omelette » qui ne sont pas précédées par
« une »
<!-- #endregion -->

```python
text = """oui très bien maintenant il reste plus que quelques petites questions pour sit-
oui c' était les c' était la dernière chose que je voulais vous demander les Anglais prétendent que même dans les moindres choses y a des différences entre euh la façon de vivre des Français et des Anglais et euh c' est pourquoi euh ils demandent euh ils se demandaient comment en France par exemple on faisait une omelette et s' il y avait des différences entre euh la façon française de faire une omelette et la façon anglaise alors si vous pouviez décrire comment vous vous faites une  omelette ?
tout d' abord on m- on casse les oeufs dans un saladier euh on mélange le le blanc et le jaune et puis on on bat avec soit avec une fourchette soit avec un appareil parce qu' il existe des appareils pour battre les oeufs euh vous prenez une poêle vous y mettez du beurre et quand il est fondu euh vous versez l' omelette par dessus euh t- j' ai oublié de dire qu' on mettait du sel et du poivre dans dans les oeufs
hm hm ?
et quand euh vous avez versé le les oeufs dans la dans la poêle euh vous euh vous quand ça prend consistance vous retournez votre omelette en faisant attention de euh de la retourner comme il faut évidemment qu' elle ne se mette pas en miettes et vous la faites cuire de l' autre côté et pour la présenter on la plie en deux maintenant on peut mettre aussi dans le dans le dans les oeufs euh des fines herbes"""
```

```python
import re
pattern = r"(?<!une )omelette"
re.findall(pattern, text)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Mais je veux pouvoir le faire même s'il y a plusieurs espaces
<!-- #endregion -->

```python
pattern = r"(?<!une\s+)omelette"
re.findall(pattern, text)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
`re` ne permet pas de faire ce genre de choses (un *lookbehind* de taille variable) mais
[`regex`](https://pypi.org/project/regex) oui
<!-- #endregion -->

```python
import regex
pattern = r"(?<!une\s+)omelette"
regex.findall(pattern, text)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## `pip`, le gestionnaire

[`pip`](https://pip.pypa.io) est le gestionnaire de paquets intégré à python (sauf sous Debian 😠).
Comme tout bon gestionnaire de paquet il sait

- Installer un paquet `pip install regex`
- Donner des infos `pip show regex`
- Le mettre à jour `pip install -U regex`
- Le désinstaller `pip uninstall regex`
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## PyPI, le *cheeseshop*

Situé à <https://pypi.org>, PyPI liste les paquets tiers — conçus et maintenus par la
communauté — pour Python.

Quand vous demandez à `pip` d'installer un paquet, c'est là qu'il va le chercher par défaut.
<!-- #endregion -->

```python
!pip search regex
```

Vous pouvez aussi le parcourir dans l'interface web, c'est un bon point de départ pour éviter de
réinventer la roue.

Le moteur de PyPI est libre et rien ne vous empêche d'héberger votre propre instance, il suffira
alors d'utiliser pip avec l'option `--index-url <url>`

<!-- #region slideshow={"slide_type": "subslide"} -->
`pip` sait faire plein d'autres choses et installer depuis beaucoup de sources. Par exemple un dépôt
git
<!-- #endregion -->

```python
!pip install --force-reinstall git+https://github.com/psf/requests.git
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Attention à utiliser **le bon** pip

Ou plutôt, c'est python qui a un souci

```bash
$ pip install --user regex
Successfully installed regex-2019.11.1
$ python3 -c "import regex;print(regex.__version__)"
ModuleNotFoundError: No module named 'regex
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
```bash
$ pip --version
pip 19.3.1 from /home/lgrobol/.local/lib/python3.8/site-packages/pip (python 3.8)
$ python3 --version
Python 3.7.5rc1
```
<!-- #endregion -->

<!-- #region -->
La solution : exiger le bon python

```bash
$ python3 -m pip install --user regex
Successfully installed regex-2019.11.1
$ python3 -c "import regex;print(regex.__version__)"
2.5.65
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## *Freeze* !

Une fonction importante de `pip` est la capacité de geler les versions des paquets installés et de
les restaurer
<!-- #endregion -->

```python
%pip freeze
```

(Traditionnellement, pour les sauvegarder : `pip freeze > requirements.txt`)

Il est **fortement** recommandé de le faire quand on lance une expérience pour pouvoir la reproduire
dans les mêmes conditions si besoin.

<!-- #region slideshow={"slide_type": "fragment"} -->
Pour restaurer les mêmes versions

```bash
pip install -r requirements.txt
```

(éventuellement avec `-U` et `--force-reinstall` en plus)

On peut aussi écrire des `requirements.txt` à la main pour préciser les dépendances d'un projet
<!-- #endregion -->

```python
!cat ../../requirements.txt
```

<!-- #region slideshow={"slide_type": "slide"} -->
## `virtualenv`

À force d'installer tout et n'importe quoi, il finit fatalement par arriver que

- Des paquets que vous utilisez dans des projets séparés aient des conflits de version
- Vous passiez votre temps à installer depuis des `requirements.txt`
- Vos `requirements.txt` soient soit incomplets, soit *trop* complets

Et arrive la conclusion
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
**En fait il me faudrait une installation différente de python pour chaque projet !**
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Et c'est précisément ce qu'on va faire
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
%pip install virtualenv
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Comment ?

- Placez-vous dans un répertoire vide

```bash
mkdir -p ~/tmp/python-crashcourse && cd ~/tmp/python-crashcourse
```

- Entrez

```bash
python3 -m virtualenv .venv
```

- Vous avez créé un environnement virtuel \o/ activez-le

```bash
source .venv/bin/activate
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Et alors ?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
```bash
$ pip list
Package    Version
---------- -------
pip        19.3.1 
setuptools 41.6.0 
wheel      0.33.6
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Alors vous avez ici une installation isolée du reste !
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
```bash
wget https://raw.githubusercontent.com/LoicGrobol/apprentissage-artificiel/main/requirements.txt
pip install -r requirements.txt
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Et maintenant vous avez tous les paquets utilisés dans ce cours. En bonus

- `pip` est le bon `pip`
- `python` est le bon `python`
- Tout ce que vous faites avec python n'agira que sur `.venv`, votre système reste propre !

En particulier, si rien ne va plus, il suffit de supprimer `.venv`
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### `virtualenv` vs `venv`

Il existe dans la distribution standard (sauf pour Debian 🙃) le module `venv` et un module tiers
`virtualenv`.

`venv` est essentiellement une version minimale de `virtualenv` avec uniquement les fonctionnalités
strictement nécessaires. En pratique on a rarement besoin de plus **sauf** quand on veut installer
plusieurs versions de Python en parallèle.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Comment travailler proprement

À partir de maintenant vous pouvez (et je vous recommande de)

- **Toujours** travailler dans un virtualenv
- **Toujours** lister vos dépendances tierces dans un requirements.txt
- **Toujours** `pip freeze`er les versions exactes pour vos expés (dans un `frozen-requirements.txt`
  par exemple.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Extensions

Quelques autres trucs utilse

- [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io) est un outil bien utile pour gérer
  vos environnements virtuels quand vous commencez à en avoir beaucoup.
- [pyenv](https://github.com/pyenv/pyenv) a une autre philosophie de travail, mais permet de faire
  de choses similaires.
- Si vous utilisez d'autres shell, il faudra parfois adapter :
  - Pour [`fish`](https://fishshell.com/), utilisez.
    [virtualfish](https://github.com/justinmayer/virtualfish/)
  - Pour [`xonsh`](https://xon.sh) (💜), utilisez
    [vox](https://xon.sh/python_virtual_environments.html).
<!-- #endregion -->
