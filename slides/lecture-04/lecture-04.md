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

[comment]: <> "LTeX: language=fr"

<!-- #region slideshow={"slide_type": "slide"} -->
Cours 4 : Modules
=================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Les fonctions c'est bien

Quand on réutilise plusieurs fois le même morceau de code, c'est pratique de ne pas avoir à se répéter
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
def un_truc_long_à_écrire(s):
    res_lst = []
    for c in s:
        if c.islower():
            res_lst.append(((ord(c) - 84) % 26) + 97)
        elif c.isupper():
            res_lst.append(((ord(c) - 52) % 26) + 65)
        else:
            res_lst.append(ord(c))
    return "".join([chr(x) for x in res_lst])
```

```python
un_truc_long_à_écrire("Arire tbaan tvir lbh hc")
```

```python
un_truc_long_à_écrire("Arire tbaan yrg lbh qbja")
```

<!-- #region slideshow={"slide_type": "subslide"} -->
C'est aussi pratique pour séparer des morceaux de code qui font des choses différentes
<!-- #endregion -->

```python
from collections import Counter

def most_common(lst, n):
    """Renvoie les n éléments les plus fréquents de lst"""
    counts = Counter(lst)
    sorted_by_freq = sorted(counts.keys(), key=counts.get, reverse=True)
    return sorted_by_freq[:n]

def keep_only_10_most_common(s):
    """Ne garder que les 10 éléments les plus communs de s"""
    keep = most_common(s, 10)
    res = []
    for c in s:
        if c in keep:
            res.append(c)
        else:
            res.append("_")
    return "".join(res)

keep_only_10_most_common("Aujourd’hui, maman est morte. Ou peut-être hier, je ne sais pas. J’ai reçu un télégramme de l’asile : « Mère décédée. Enterrement demain. Sentiments distingués. » Cela ne veut rien dire. C’était peut-être hier.")
```

<!-- #region slideshow={"slide_type": "subslide"} -->
## Pour vivre heureux, cachons le code

Cette division du code en morceaux plus petits et autonomes s'appelle *séparation des préoccupations*.

**Principe** : chaque fonction doit faire une chose et une seule en étant la plus générique possible.

Par exemple, peu importe que je n'applique `most_common` que sur des chaînes de caractères ici, elle marcherait pour n'importe quel itérable
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
- Je ne m'occupe que d'une chose à la fois
- Je ne m'encombre pas l'esprit avec des informations qui ne concernent pas cette chose
- Quand j'utilise ma fonction, je ne me soucie plus de comment elle a été écrite (et l'implémentation est donc facile à changer)
- Accessoirement, je ne pollue pas l'espace de nom avec des variables qui ne serviront plus

On rejoint le concept d'API, dont on reparlera
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Les modules

1. Ouvrez votre éditeur de texte préféré
2. Créez un nouveau fichier (dans un dossier vide, pour la suite)
3. Enregistrez le sous le nom `libspam.py`
4. Voilà, vous avez créé un module
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Qu'est ce qu'un module ?

Techniquement, n'importe quel fichier portant l'extension `.py` et ne comprenant que du code interprétable par Python est un module.

Jusque là ça n'a pas l'air très intéressant.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
1. Dans votre fichier `libspam.py`, insérez le code suivant
  ```python
  def sing():
        print("spam, spam, lovely spam!")
  ```
2. Créez un fichier `spam.py` **dans le même dossier** et insérez-y
  ```python
  import libspam
  libspam.sing()
  ```
3. Exécutez `spam.py` (`python3 spam.py`)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Pourquoi ?

C'est le niveau suivant de séparation des préoccupations : du code autonome dans un fichier
différent

→ Non seulement vous n'avez pas besoin de **penser** au code mais vous n'avez même pas à le **voir**

Vous pouvez même garder les mêmes modules d'un projet à l'autre, plus de copier-coller brutal de
code entre vos projets !

Mieux : vous pouvez plus facilement partager du code avec d'autres et utiliser le leur (on y
reviendra)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Utiliser des modules

Vous utilisez depuis longtemps des modules : ceux de la bibliothèque standard par exemple
<!-- #endregion -->

```python
import re

re.sub(r"[^aàæeéèêëiîïoôœuûüyÿ]", "", "longtemps je me suis couché de bonne heure")
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Du point de vue *interne* (dans votre code), un module est un *objet* qui apparaît dans votre code
grâce à une invocation de `import`
<!-- #endregion -->

```python
import sys

type(sys)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
C'est un objet essentiellement normal, qui possède des propriétés et des méthodes, qui sont celles
définies dans le fichier `.py` correspondant
<!-- #endregion -->

```python
dir(re)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Par convention, ici comme ailleurs, les membres à usage privé (que vous n'êtes pas censés utilisés
commencent par un underscore
<!-- #endregion -->

```python
[m for m in dir(re) if m.startswith("_")]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Les membres entourés d'underscores (comme `__file__`), ou *dunders* sont des membres traités par le
langage de façon particulière.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
re.__file__
```

Par exemple `__file__` n'est pas défini dans `re.py` mais est affecté au moment de `import re`

<!-- #region slideshow={"slide_type": "slide"} -->
# `import`

La commande `import` est l'une des plus importantes commandes de python. Quand elle est invoquée
comme `import machin`, Python

1. Cherche dans les dossiers de modules un fichier nommé `machin.py`
2. Exécute le code qu'il contient
3. Rend disponible les objets qui en résultent en les leur donnant le nom `machin.<bidule>` où
   `<bidule>` est le nom de l'objet dans `machin`

Autrement dit, si vous avec `truc = 1` dans `machin.py`, quand vous importez `machin`, vous avec
`machin.truc` avec la valeur `1`.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Importer des modules

**Où** Python va chercher les modules et comment il les nomme est un sujet complexe et on ne
traitera pas tout, cependant

- Les modules qui se trouvent dans le même dossier que votre script sont directement importables
- Les modules présent dans le `PYTHONPATH` sont directement importables
<!-- #endregion -->

```python
import sys
sys.path
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut également importer des modules qui se trouvent dans des sous-dossiers du dossier de script.

Si par exemple vous avez l'arborescence suivante

```
.
├── script.py
└── spam
    ├── ham.py
    └── __init__.py
```

Alors `ham.py` peut être importé dans `script.py` en utilisant la commande

```python
import spam.ham
```

Et sera disponible sous le nom `spam.ham`. Par exemple dans la bibliothèque standard, vous trouverez
`sys.path` sur ce modèle.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Imports avancés

Si vous n'aimez pas le nom d'un module, vous pouvez l'importer sous un autre nom avec `import … as
…`
<!-- #endregion -->

```python
import numpy as np

np.max([1,7,3])
```

Ce qui peut être utile pour les noms à rallonge

```python
import matplotlib.pyplot as plt
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Vous pouvez aussi n'importer que ce que vous intéresse avec `from … import …`
<!-- #endregion -->

```python
from re import sub
sub(r"[aeiou]", "💓", "Plurital")
```
Ce qui concerne à la fois les membres des modules et les sous-modules.

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut aussi importer l'intégralité d'un module
<!-- #endregion -->

```python
from re import *
sub(r"[aeiou]", "💓", "Plurital")
```

On le trouve souvent dans la nature mais c'est en général une **très mauvaise idéé**:

- Ça rend très difficile de savoir d'où viennent les objets dans votre module
- En ajoutant les fonctions dans l'espace de nommage du script vous pouvez écraser des fonctions
  existantes.

Et finalement :

```python
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()) as ham:
    import this
print(ham.getvalue().splitlines()[3])
```

## Un package

```python
! tree operations_pack
```

Un package python peut contenir des modules, des répertoires et sous-répertoires, et bien souvent du non-python : de la doc html, des données pour les tests, etc…

Le répertoire principal et les répertoires contenant des modules python doivent contenir un fichier `__init__.py`

`__init__.py` peut être vide, contenir du code d'initialisation ou contenir la variable `__all__`

```python
import operations_pack.simple
operations_pack.simple.addition(2, 4)
```

```python
from operations_pack import simple
simple.soustraction(4, 2)
```

``__all__`` dans ``__init__.py`` définit quels seront les modules qui seront importés avec ``import *``

```python
from operations_pack.avance import *
multi.multiplication(2,4)
```

# Pas de main en Python ?

Vous trouverez fréquemment le test suivant dans les scripts Python :

```python
if __name__ == '__main__':
    instruction1
    instruction2
```

ou

```python
def main():
    instruction

if __name__ == '__main__':
    main()
```

Cela évite que le code sous le test ne soit exécuté lors de l'import du script : `__name__` est une
variable créée automatiquement qui vaut `__main__` si le script a été appelé en ligne de commande,
le nom du script s'il a été importé.

Accessoirement cela permet d'organiser son code et de le rendre plus lisible
Cela permet aussi d'importer les fonctions du script à la manière d'un module

Je vous ~recommande vivement~ demande de l'inclure dans tous vos scripts. On verra aussi d'autres
façons de gérer les interfaces en ligne de commande…

## Où sont les modules et les packages ?

Pour que `import` fonctionne il faut que les modules soient dans le PATH.

```python
import sys
sys.path
```

``sys.path`` est une liste, vous pouvez la modifier **mais évitez à moins d'avoir une très bonne
raison**.

```python
sys.path.append("[...]") # le chemin vers le dossier operations_pack
sys.path
```