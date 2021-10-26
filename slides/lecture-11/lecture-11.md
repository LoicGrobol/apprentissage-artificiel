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


On va commencer par utiliser [`gensim`](https://radimrehurek.com/gensim), qui nous fournit plein de
modèles tout faits.

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

### 🧨 Exo 🧨

1\. Essayez avec d'autres mots. Quels semblent être les critères qui font que des mots sont proches
dans ce modèle.

2\. Comparer avec les vecteurs du modèle `"glove-twitter-100"`. Y a-t-il des différences ?
**Note** : il peut être long à télécharger, commencez par ça.

3\. Entraîner un modèle [`word2vec`](https://radimrehurek.com/gensim/models/word2vec.html) avec
gensim sur les documents du dataset 20newsgroup. Comparer les vecteurs obtenus avec les précédents.

## Sémantique lexicale distributionnelle

### Principe général

Pour le dire vite :

La *sémantique lexicale*, c'est l'étude du sens des mots. Rien que dire ça, c'est déjà faire
l'hypothèse hautement non-triviale que les mots ont un (ou plus vraisemblablement des) sens.

C'est tout un pan de la linguistique et on ne rentrera pas ici dans les détails (mêmes s'il sont
passionnants !) parce que notre objectif est *applicatif* :

- Comment représenter le sens d'un mot ?
- Peut-on, à partir de données linguistiques, déterminer le sens des mots ?
- Et plus tard : comment on peut s'en servir ?

Une façon de traiter le problème, c'est de recourir à de l'annotation manuelle (par exemple avec
[Jeux de mots](http://www.jeuxdemots.org), d'ailleurs, vous avez joué récemment ?).

On ne se penchera pas plus dessus ici : ce qui nous intéresse, c'est comment traiter ce problème
avec de l'apprentissage, et en particulier avec de l'apprentissage sur des données non-annotées.

Pour ça, la façon la plus populaire (et pour l'instant celle qui semble la plus efficace) repose sur
l'**hypothèse distributionnelle**, formulée ainsi par Firth

> You shall know a word by the company it keeps.

Autrement dit : des mots dont le sens est similaire devraient apparaître dans des contextes
similaires et vice-versa.


Si on pousse cette hypothèse à sa conclusion naturelle : on peut représenter le sens d'un mot par
les contextes dans lesquels il apparaît.

Le principal défaut de cette vision des choses, c'est que ce n'est pas forcément très interprétable,
contrairement par exemple à des représentations en logique formelle. Mais ça nous donne des moyens
très concrets d'apprendre des représentations de mots à partir de corpus non-annotés.

### Modèle par documents

Par exemple une façon très simple de l'appliquer, c'est de regarder dans quels documents d'un grand
corpus apparaît un mot : des mots qui apparaissent dans les mêmes documents avec des fréquences
similaires devraient avoir des sens proches.

Qu'est-ce que ça donne en pratique ? Et bien souvenez-vous du modèle des sacs de mots : on peut
représenter des documents par les fréquences des mots qui y apparaissent. Ça nous donne une
représentation vectorielle d'un corpus sous la forme d'une matrice avec autant de ligne que de
documents, autant de lignes que de mots dans le vocabulaire et où chaque cellule est une fréquence.

Jusque-là on s'en est servi en lisant les lignes pour récupérer des représentations vectorielles des
documents, mais si on regarde les colonnes, on récupère des **représentations vectorielles des
mots** !

(Ce qui répond à la première question : comment représenter le sens ? Comme le reste, avec des
vecteurs !)

### 🐢 Exo 🐢

À partir du corpus 20newsgroup, construire un dictionnaire associant chaque mot du vocabulaire à une
représentation vectorielle donnant ses occurrences dans chacun des documents du corpus.

**N'hésitez pas à recycler du code**

Est-ce que les distances entre les vecteurs de mots ressemblent à celles qu'on observait avec Gensim ?

Est-ce que vous voyez une autre façon de récupérer des vecteurs de mots en utilisant ce corpus ?


### Cooccurrences

Une autre possibilité, plutôt que de regarder dans quels documents apparaît un mot, c'est de regarder directement les autres mots dans son voisinage. Autrement dit les cooccurrences.

L'idée est la suivante : on choisit un paramètre $n$ (la « taille de fenêtre ») et on regarde pour
chaque mot du corpus les $n$ mots précédents et les $n$ mots suivants. Chacun de ces mots voisins
constitue une cooccurrence. Par exemple avec une fenêtre de taille $2$, dans 

> Le petit chat est content

On a les cooccurrences `("le", "petit")`, `("le", "chat")`, `("petit", "chat")`, `("petit", "est")`…

Comment on se sert de ça pour récupérer une représentation vectorielle des mots ? Comme
d'habitude : on compte ! Ici on représentera chaque mot par un vecteur avec autant de coordonnées
qu'il y a de mots dans le vocabulaire, et chacune de ces coordonnées sera le nombre de cooccurrences
avec le mot correspondant.


### 🦘 Exo 🦘

À partir du corpus 20newsgroup, construire un dictionnaire associant chaque mot du vocabulaire à une
représentation vectorielle par la méthode des cooccurrences pour une taille de fenêtre choisie.

Est-ce que les distances entre les vecteurs de mots ressemblent à celles qu'on observait avec les
représentations précédentes ?

## Extensions

Le défaut principal de ces représentations, c'est qu'elles sont très **creuses** : beaucoup de
dimensions, mais qui contiennent surtout des zéros. Ce n'est pas très économique à manipuler et
c'est moins utile quan on veut les utiliser comme entrée pour des systèmes de TAL, comme des réseaux
de neurones

L'essentiel du travail fait ces dix dernières années dans ce domaine consiste à trouver des
représentations **denses** : moins de dimensions (au plus quelques centaines) mais peu de zéros. ON
parle alors en français de *plongements* et en anglais de *word embeddings*.

Il y a beaucoup de façons de faire ça, *Speech and Language Processing* détaille la plus connue,
*word2vec* et je vous encourage à aller voir comment ça marche.

Une autre possibilité d'extensions est de descendre en dessous de l'échelle du mot, et d'utiliser
des sous-mots, qui peuvent éventuellement avoir un sens linguistique (comme des morphèmes), mais
sont eux aussi en général appris de façon non-supervisée. C'est ce que fait
[FastText](https://fasttext.cc/docs/en/python-module.html), qui est plus ou moins ce qui se fait de
mieux en termes de représentations vectorielles de mots.

## 👽 Exo 👽

(Pour les plus motivé⋅e⋅s, mais la doc vous dit déjà presque tout)

1\. Entraîner un modèle non-supervisé [`FastText`](https://fasttext.cc/docs/en/python-module.html)
sur 20 newsgroups et voir si les similarités sont les mêmes que pour les modèles précédents.

2\. Entraîner et tester un modèle de classification FastText sur 20 newsgroup.