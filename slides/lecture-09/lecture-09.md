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

[comment]: <> "LTeX: language=fr"

<!-- #region slideshow={"slide_type": "slide"} -->
Cours 9 : Classification de documents partie 1, *Naïve Bayes* et Régression Logistique
======================================================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-11
<!-- #endregion -->

```python
from IPython.display import display
```

## Pitch

La **classification de documents** est une tâche de TAL qui consiste à ranger un document dans une et
une seule classe parmi un ensemble prédéfini.

Elle est très importante historiquement, car elle a été une des passerelles entre l'informatique (où
on peut la voir comme un cas particulier de la tâche générale de classification) et le TAL. En
pratique, comme beaucoup des tâches de TAL peuvent se reformuler comme des tâches de classification,
elle est aussi d'une importance cruciale.

En ce qui nous concerne, elle est aussi intéressante parce que les techniques classiques de
classification par apprentissage vont nous donner l'occasion de (re?)découvrir des concepts qui vont
nous servir dans toute la suite de ce cours.
On va l'aborder au travers de deux techniques : *Naïve Bayes* (le « modèle bayésien naïf » 🙄) et la
régression logistique, appliquées au modèle de représentation des documents par **sacs de mots**.

On se basera pour la théorie et les notations sur les chapitres 4 et 5 de [*Speech and Language
Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et James H. Martin, qu'il
est donc bon de garder à portée de main.

Pour éviter d'avoir à prédater des données, on va se servir du [dataset d'exemple de `scikit-learn`
*20
newsgroups*](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
qu'on a [déjà](../lecture-07/lecture-07.md#Classification-de-textes) rencontré, en revanche on évitera de se servir directement des fonctions de `scikit-learn`. On sait déjà faire et l'objectif ici est de le faire à la mano pour bien comprendre ce qui se passe. On se servira aussi
pas mal de NumPy, n'hésitez donc pas à aller revoir [le cours qui le
concerne](../lecture-06/lecture-06.md).

**C'est parti !**

```python
%pip install -U numpy scikit-learn
```

```python
import numpy as np
```

## Le dataset

```python
from sklearn.datasets import fetch_20newsgroups

categories = [
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
]

data_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
)

data_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
)
```

```python
print(data_train.DESCR)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Bon, voyons ce que ce truc a concrètement dans le ventre
<!-- #endregion -->

```python
print(len(data_train.data))
print(len(data_test.data))
```

```python
data_train.keys()
```

```python
type(data_train.data)
```

```python
data_train.data[0]
```

```python
type(data_train.target)
```

```python
data_train.target[0]
```

```python
data_train.target_names[0]
```

Bon, c'est plutôt clair

## Sac de mots


On va commencer par transformer ces documents en sacs de mots. Pour ça on [recycle](../lecture-08/lecture-08.md)

```python
import re
def poor_mans_tokenizer_and_normalizer(s):
    # Cette fois-ci on vire les nombres, les signes de ponctuation et les trucs bizarres
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]
```

```python
from collections import Counter

def get_counts(doc):
    return Counter(poor_mans_tokenizer_and_normalizer(doc))

get_counts(data_train.data[0])
```

Maintenant il faut le faire pour tous les docs 🏹

```python
bows = [get_counts(doc) for doc in data_train.data]
```

Et il nous faut récupérer tout le vocabulaire

```python
vocab = sorted(set().union(*bows)) # Pourquoi `sorted` à votre avis ?
len(vocab)
```

Et pour rendre le tout facile à manipuler on va en en faire un tableau NumPy de la forme `len(train_data)×len(vocab)` qui
tel que le contenu de la cellule `(i, j)` soit le nombre d'occurrence du mot `i` dans le document `j`. On a [déjà](../lecture-06/lecture-06.md#%F0%9F%91%9C-Exo%E2%80%AF:-les-sacs-de-mots-%F0%9F%91%9C) fait ça.


On commence par faire un dict avec le vocabulaire

```python
w_to_i = {w: i for i, w in enumerate(vocab)}
```

```python
bow_array = np.zeros((len(bows), len(vocab)))
for i, bag in enumerate(bows):
    for w, c in bag.items():
        bow_array[i, w_to_i[w]] = c
```

On peut aussi faire comme ça mais c'est **beaucoup** plus lent. Est-ce que vous pouvez devinez pourquoi ?

```python
# bow_array = np.array(
#     [
#         [bag[w] for w in vocab]
#         for bag in bows
#     ]
# )
```

```python
bow_array
```

## Le modèle *Naïve Bayes*

À suivre au tableau !

## 🧙🏻 Exo 🧙🏻

À vous de bosser maintenant. Écrivez

1\. Une fonction qui prend en argument un tableau comme `data_train.target` et qui renvoie un
tableau `class_probs` tel que `class_probs[c]` soit $P(c)$. On choisira le modèle de vraisemblance
maximal, soit ici simplement celui qui utilise pour probabilités les fréquences empiriques $P(c) =
\frac{\text{nombre d'occurrences de $c$}}{\text{taille de l'échantillon}}$.

```python
def get_class_probs(target):
    pass  # Votre code ici
```

2\. Une fonction qui prend en argument un tableau comme `bow_array` et renvoie un tableau
`word_probs` tel que `word_probs[c][w]` soit $P(w|c)$. On utilise toujours le modèle de
vraisemblance maximal mais en considérant des occurrences booléenne et un lissage laplacien :
$P(w|c)=\frac{\text{nombre de documents de $c$ dans lesquels $w$ apparaît} + 1}{\text{nombre de mots
dans l'ensemble des documents de $c$}+\text{taille du vocabulaire}}$.

N'hésitez pas à écrire des boucles, au moins pour commencer, avant de passer à du NumPy fancy.

```python
def get_word_probs(bows):
    pass  # Votre code ici
```

Voilà, on a un modèle de classification *Naïve Bayes* 👏🏻


Il reste à savoir comment s'en servir. Je vous laisse coder ça vous-mêmes. N'hésitez pas à faire des
fonctions auxiliaires.


3\. Une fonction qui prend en argument un document et renvoie la classe la plus probable notre modèle. Pensez à travailler en log-probabilités

```python
def predict_class(doc):
    pass  # Votre code cic
```

Vous pouvez maintenant évaluer le modèle en calculant son exactitude sur l'ensemble de test. 


4\. Un script qui entraîne le modèle et le sauvegarde (sous la forme qui vous paraît la plus
appropriée) et un qui charge le modèle et prédit la classe de chacun des documents d'un corpus.


Courage, c'est pour votre bien. Si vous vous ennuyez ça peut être le bon moment pour découvrir [click](https://click.palletsprojects.com/en/8.0.x/).

5\. Comme ultime raffinement

## Classifieur logistique

## 🧠 Exo 🧠

1\. Tracer avec matplotlib la courbe représentative de la fonction logistique.

2\. À l'aide d'un lexique de sentiment, écrivez un classifieur logistique à deux features : nombre
de mots positifs et nombre de mots négatifs avec les poids respectifs $0.6$ et $0.4$ et pas de terme
de biais. Appliquez ce classifieur sur le mini-corpus LMDB et calculez son exactitude.

## Apprendre un classifieur logistique

## 😩 Exo 😩

1\. Coder la log-vraisemblance négative et son gradient

2\. S'en servir pour apprendre les poids à donner aux features précédentes à l'aide du mini-corpus LMDB
