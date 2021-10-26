---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

<!-- #region slideshow={"slide_type": "slide"} -->
Cours 11 : Représentations lexicales vectorielles
=================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-27
<!-- #endregion -->

```python
from IPython.display import display
```

## Représentaquoi ?

**Représentations lexicales vectorielles**, ou en étant moins pédant⋅e « représentations
vectorielles de mots ». Comment on représente des mots par des vecteurs, quoi.


Mais qui voudrait faire ça, et pourquoi ?


Tout le monde, et pour plein de raisons


On va commencer par utiliser [`gensim`](https://radimrehurek.com/gensim), qui nous fournit plein de modèles tout faits.

```python
%pip install -U gensim
```

et pour démarrer, on va télécharger un modèle tout fait

```python
import gensim.downloader as api
wv = api.load("glove-wiki-gigaword-50")
```

OK, super, qu'est-ce qu'on a récupéré ?

```python
type(wv)
```

C'est le bon moment pour aller voir [la
doc](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors).
On y voit qu'il s'agit d'un objet associant des mots à des vecteurs.

```python
wv["monarch"]
```

Des vecteurs stockés comment ?

```python
type(wv["monarch"])
```

Ah parfait, on connaît : c'est des tableaux numpy

```python
wv["king"]
```

```python
wv["queen"]
```

D'accord, très bien, on peut faire quoi avec ça ?


Si les vecteurs sont bien faits (et ceux-ci le sont), les vecteurs de deux mots « proches »
devraient être proches, par exemple au sens de la similarité cosinus

```python
import numpy as np
def cosine_similarity(x, y):
    """Le cosinus de l'angle entre `x` et `y`."""
    return np.inner(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
```

```python
cosine_similarity(wv["monarch"], wv["king"])
```

```python
cosine_similarity(wv["monarch"], wv["cat"])
```

En fait le modèle nous donne directement les mots les plus proches en similarité cosinus.

```python
wv.most_similar(["monarch"])
```

Mais aussi les plus éloignés

```python
wv.most_similar(negative=["monarch"])
```

### Exo

1\. Essayez avec d'autres mots. Quels semblent être les critères qui font que des mots sont proches
dans ce modèle.

2\. Comparer avec les vecteurs du modèle `"glove-twitter-100"`. Y a-t-il des différences ?
**Note** : il peut être long à télécharger, commencez par ça.

3\. Entraîner un modèle [`word2vec`](https://radimrehurek.com/gensim/models/word2vec.html) avec
gensim sur les documents du dataset 20newsgroup. Comparer les vecteurs obtenus avec les précédents.

## Sémantique lexicale

## Word2vec

## Exploration

ICI on parle d'algèbre entre mots

jurafsky 6.10 et 11

## TP Fasttext

(dans un autre notebook)

- Faire un tuto rapide pour https://fasttext.cc/docs/en/python-module.html
- TODO : Apprendre et tester des word embeddings sur un autre dataset (genre 20 newsgroup)
- Tester d'apprendre un modèle de classif dans sklearn en utilisant des embeddings fasttext, tester plusieurs classifieurs
- Comparer avec un modèle de classif apris dans fasttext
