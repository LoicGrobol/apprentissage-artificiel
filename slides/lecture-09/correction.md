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
Correction 9 : *Naïve Bayes*
=======================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-11
<!-- #endregion -->

```python
from IPython.display import display
```

```python
%pip install -U numpy scikit-learn
```

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
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

Maintenant, il faut le faire pour tous les docs 🏹

```python
bows = [get_counts(doc) for doc in data_train.data]
```

Et il nous faut récupérer tout le vocabulaire

```python
vocab = sorted(set().union(*bows)) # Pourquoi `sorted` à votre avis ?
len(vocab)
```

Et pour rendre le tout facile à manipuler on va en en faire un tableau NumPy de la forme
`len(train_data)×len(vocab)` qui tel que le contenu de la cellule `(i, j)` soit le nombre
d'occurrences du mot `i` dans le document `j`. On a
[déjà](../lecture-06/lecture-06.md#%F0%9F%91%9C-Exo%E2%80%AF:-les-sacs-de-mots-%F0%9F%91%9C) fait
ça.


On commence par faire un dict avec le vocabulaire

```python
w_to_i = {w: i for i, w in enumerate(vocab)}
```

```python
bow_array = np.zeros((len(bows), len(vocab)))
for i, bag in enumerate(bows):
    for w, c in bag.items():
        bow_array[i, w_to_i[w]] = c
bow_array
```

## 🧙🏻 Exo 🧙🏻

> À vous de bosser maintenant. Écrivez
>
> 1\. Une fonction qui prend en argument un tableau comme `data_train.target` et qui renvoie un
> tableau `class_probs` tel que `class_probs[c]` soit $P(c)$. On choisira le modèle de vraisemblance
> maximal, soit ici simplement celui qui utilise pour probabilités les fréquences empiriques $P(c) =
> \frac{\text{nombre d'occurrences de $c$}}{\text{taille de l'échantillon}}$.

```python
def get_class_probs(target):
    # Chaque élément est un numéro de classes, on va supposer que toutes les classes y sont 
    # représentées
    # Les classes vont aller de 0 à n inclus, il y en a donc n+1
    n_classes = np.max(target) + 1
    # On va compter les éléments de chaque classe
    # On pourrait aussi faire avec `Counter()` mais comme on est parti avec des arrays, on va rester 
    # avec des arrays
    counts = np.zeros(n_classes)
    for c in target:
        counts[c] += 1
    # On normalise par la somme pour avoir les fréquences empiriques
    total = counts.sum()
    # IL Y A UN BROADCAST ICI, REGARDEZ LES DIMENSIONS DES TABLEAUX
    return counts/total
```

On teste

```python
class_probs = get_class_probs(data_train.target)
class_probs
```

On visualise ?

```python
plt.bar(np.arange(class_probs.shape[0]), class_probs)
plt.xlabel("classe")
plt.ylabel("proportion")
plt.title("répartition des classes") 
plt.show()
```

Voilà un dataset bien équilibré


On aurait aussi pu faire directement

```python
def get_class_probs(target):
    return np.bincount(target)/target.size
get_class_probs(data_train.target)
```

> 2\. Une fonction qui prend en argument un tableau comme `bow_array` et un tableau de classes
> `target` comme précédemment et renvoie un tableau `word_probs` tel que `word_probs[c][w]` soit
> $P(w|c)$. On utilise toujours le modèle de vraisemblance maximal mais avec un **lissage
> laplacien** : $P(w|c)=\frac{\text{nombre d'occurences de $w$ dans $c$} + 1}{\text{nombre de mots
> dans l'ensemble des documents de $c$}+\text{taille du vocabulaire}}$.
>
> N'hésitez pas à écrire des boucles, au moins pour commencer, avant de passer à du NumPy fancy.


On va commencer par le faire par étapes pour visualiser puis on mettra tout dans une fonction


On construit d'abord les comptes d'occurrences et on normalisera après


`bow_array` est de dimensions (nombre de documents, nombre de mots)

```python
bow_array.shape
```

donc

```python
vocabulary_size = bow_array.shape[1]
```

Comme précédemment, `data_train.target` contient les comptes des classes dont il y a

```python
n_classes = np.max(data_train.target)+1
```

On va stocker les comptes dans un tableau de dimension `(nombre de classes, nombre de mots)`, on va
d'abord le créer puis on le remplira dans une boucle.

```python
counts = np.zeros((n_classes, vocabulary_size))
counts
```

Ah, mais on a dit qu'on allait ajouter $1$ à tous les comptes pour lisser, donc il nous faut en fait

```python
counts = np.ones((n_classes, vocabulary_size))
counts
```

Maintenant on remplit avec une boucle : on itère sur les sacs de mots et on ajoute leurs comptes
d'occurrences à la ligne qui correspond à leur classe. C'est assez facile : le nombre d'occurrence
de chaque mot dans un doc, c'est juste sa ligne dans `bow_array`

```python
bow_array[0]
```

On va vouloir ajouter tous ces comptes à la ligne correspondant à sa classe (donc
`data_train.targets[0]`) dans `counts`. On va pouvoir le faire directement avec la fonction
d'addition des arrays

```python
bow_array[0] + counts[data_train.target[0]]
```

On fait ça pour tous les documents :

```python
for doc, c in zip(bow_array, data_train.target):
    counts[c] += doc
counts
```

Il reste à normaliser. Pour ça, il nous faut d'abord le nombre total de mots dans chaque classe. Par
exemple pour la classe `0`

```python
np.sum(counts[0])
```

On normalise avec ça les occurrences pour cette classe

```python
counts[0]/np.sum(counts[0])
```

On le fait pour tout le monde

```python
# On part d'un tableau vide : inutile de le préremplir puisqu'on va remplir chaque ligne à la main
word_probs = np.empty((n_classes, vocabulary_size))
# `enumerate` pour savoir quelle ligne remplir
for i, class_count in enumerate(counts):
    word_probs[i] = counts[i]/np.sum(counts[i])
word_probs
```

Et voilà, on a fini !


On peut faire plus efficace avec des opérations sur les tableaux. On peut aussi passer cette section
et y revenir plus tard !

On peut directement compter les totaux par classe avec le paramètre `axis` de `numpy.sum` qui dit
sur quelle dimension on somme.

```python
total_per_class = np.sum(counts, axis=1)
total_per_class
```

On a ensuite envie de diviser chaque ligne par sa somme en utilisant le broadcast

```python tags=["raises-exception"]
counts/total_per_class
```

Mais ça ne marche pas, parce que numpy ne veut pas ajouter de dimension pour nous. On peut s'en
sortir de deux façons :


En ajoutant un axe à la somme

```python
display(total_per_class[:, np.newaxis].shape)
display(counts/total_per_class[:, np.newaxis])
```

Ou en conservant l'axe $1$ lors de la somme

```python
total_per_class = np.sum(counts, axis=1, keepdims=True)
display(total_per_class.shape)
display(counts/total_per_class)
```

On pourrait aussi remplacer la première boucle par
[`np.put_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis), mais
les boucles, c'est pas si pire.

On met tout dans une fonction

```python
def get_word_probs(bows, target):
    counts = np.ones((np.max(target)+1, bows.shape[1]))
    for doc, c in zip(bows, target):
        counts[c] += doc
    total_per_class = np.sum(counts, axis=1, keepdims=True)
    return counts/total_per_class
get_word_probs(bow_array, data_train.target)
```

Voilà, on a un modèle de classification *Naïve Bayes* 👏🏻

> 3\. Une fonction qui prend en argument un document et renvoie la classe la plus probable notre
> modèle. Pensez à travailler en log-probabilités

On va s'inspirer de l'algo 4.2 de *Speech and Language Procseeing* en le compactifiant un peu.


D'abord on calcule les log-paramètres du modèle

```python
log_prior = np.log(get_class_probs(data_train.target))
log_likelihood = np.log(get_word_probs(bow_array, data_train.target))
display(log_prior)
display(log_likelihood)
```

Et on se rappelle qu'on a déjà le vocabulaire dans `w_to_i`


Ensuite on les exploite

```python
def predict_class(doc):
    # D'abord on récupère sa représenation en sacs de mots
    bow_dict = get_counts(doc)
    bow = np.zeros(len(w_to_i))
    for w, c in bow_dict.items():
        # On ne garde que les mots connus (on peut économiser un lookup, à votre avis comment ?)
        if w not in w_to_i:
            continue
        bow[w_to_i[w]] = c
    class_likelihoods = np.empty(n_classes)
    for i, (class_log_prior, class_log_likelihood) in enumerate(zip(log_prior, log_likelihood)):
        acc = 0.0
        for word_count, word_log_likelihood in zip(bow, class_log_likelihood):
            acc += word_count*word_log_likelihood
        class_likelihoods[i] = acc + class_log_prior
    return np.argmax(class_likelihoods)
```

```python
print(data_train.data[0][:300])
predicted = predict_class(data_train.data[0])
print(f"La classe prédite pour le premier document est {predicted}, soit {data_train.target_names[predicted]}")
print(f"La classe correcte était {data_train.target[0]}, soit npdependency{data_train.target_names[data_train.target[0]]}")
```

Ou en plus rapide et compact avec un [produit scalaire](https://numpy.org/doc/stable/reference/generated/numpy.inner) vecteur-vecteur

```python
def predict_class(doc):
    bow_dict = get_counts(doc)
    bow = np.zeros(len(w_to_i))
    for w, c in bow_dict.items():
        if w not in w_to_i:
            continue
        bow[w_to_i[w]] = c
    class_likelihoods = np.empty(n_classes)
    for i, (class_log_prior, class_log_likelihood) in enumerate(zip(log_prior, log_likelihood)):
        class_likelihoods[i, ...] = class_log_prior + np.inner(bow, class_log_likelihood)
    return np.argmax(class_likelihoods)
```

Ou encore plus rapide et compact avec une [multiplication matrice-vecteur](https://numpy.org/doc/stable/reference/generated/numpy.matmul)

```python
def predict_class(doc):
    bow_dict = get_counts(doc)
    bow = np.zeros(len(w_to_i))
    for w, c in bow_dict.items():
        if w not in w_to_i:
            continue
        bow[w_to_i[w]] = c
    class_likelihoods = np.matmul(log_likelihood, bow) + log_prior
    return np.argmax(class_likelihoods)
```

Si vous voulez vous motiver à mieux apprendre numpy, je vous recommande de chronométrer les temps d'exécution de la suite avec les différentes versions de cette fonction.


> Vous pouvez maintenant évaluer le modèle en calculant son exactitude sur l'ensemble de test.


On va déjà le faire sur l'ensemble de train

```python
predictions = np.array([predict_class(text) for text in data_train.data])
display(predictions)
correct = predictions == data_train.target
display(correct)
print(f"Il y a {correct.sum()} exemples bien classés parmis {correct.size}, soit {correct.sum()/correct.size:.2%} d'exactitude")
```

Plutôt encourageant ! on va écrire une fonction pour le faire sur n'importe quel ensemble de textes

```python
def evaluate(documents, target):
    predictions = np.array([predict_class(text) for text in documents])
    correct = predictions == target
    correct
    return correct.sum()/correct.size
```

```python
evaluate(data_test.data, data_test.target)
```

> 4\. Un script qui entraîne le modèle et le sauvegarde (sous la forme qui vous paraît la plus
> appropriée) et un qui charge le modèle et prédit la classe de chacun des documents d'un corpus.


> Courage, c'est pour votre bien. Si vous vous ennuyez ça peut être le bon moment pour découvrir [click](https://click.palletsprojects.com/en/8.0.x/).


Voir [`nb_script.py`](nb_script.py)
