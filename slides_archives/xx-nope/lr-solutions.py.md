---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
TP 5 : Régression logistique
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

```python
from IPython.display import display, Markdown
```

```python
import numpy as np
```

## Vectorisations arbitraires de documents

On a vu des façons de traiter des documents vus comme des sacs des mots en les représentant comme
des vecteurs dont les coordonnées correspondaient à des nombres d'occurrences.

Mais on aimerait — entre autres — pouvoir travailler avec des représentations arbitraires, on peut
par exemple imaginer vouloir représenter un document par ŀa polarité (au sens de l'analyse du
sentiment) de ses mots.

## 🧠 Exo 🧠

### 1. Vectoriser un document

> À l'aide du lexique [VADER](https://github.com/cjhutto/vaderSentiment) (vous le trouverez aussi
dans [`data/vader_lexicon.txt`](data/vader_lexicon.txt)), écrivez une fonction qui prend en entrée
> un texte en anglais et renvoie sa représentation sous forme d'un vecteur de features à deux
> traits : polarité positive moyenne (la somme des polarités positives des mots qu'il contient
> divisée par sa longueur en nombre de mots) et polarité négative moyenne.


On commence par recycler notre tokenizer/normaliseur

```python
import re

def crude_tokenizer_and_normalizer(s):
    tokenizer_re = re.compile(
        r"""
        (?:                   # Dans ce groupe, on détecte les mots
            \b\w+?\b          # Un mot c'est des caractères du groupe \w, entre deux frontières de mot
            (?:               # Éventuellement suivi de
                '             # Une apostrophe
            |
                (?:-\w+?\b)*  # Ou d'autres mots, séparés par des traits d'union
            )?
        )
        |\S        # Si on a pas détecté de mot, on veut bien attraper un truc ici sera forcément une ponctuation
        """,
        re.VERBOSE,
    )
    return tokenizer_re.findall(s.lower())
```

On lit le lexique

```python
def read_vader(vader_path):
    res = dict()
    with open(vader_path) as in_stream:
        for row in in_stream:
            word, polarity, *_ = row.lstrip().split("\t", maxsplit=2)
            res[word] = float(polarity)
    return res
lexicon = read_vader("data/vader_lexicon.txt")
lexicon
```

Et voilà comment on récupère la représentation d'un document

```python
def featurize(text, lexicon):
    words = crude_tokenizer_and_normalizer(text)
    features = np.empty(2)
    # Le max permet de remonter les polarités négatives à 0
    features[0] = sum(max(lexicon.get(w, 0), 0) for w in words)/len(words)
    features[1] = sum(max(-lexicon.get(w, 0), 0) for w in words)/len(words)
    return features
```

On teste ?

```python
doc = "I came in in the middle of this film so I had no idea about any credits or even its title till I looked it up here, where I see that it has received a mixed reception by your commentators. I'm on the positive side regarding this film but one thing really caught my attention as I watched: the beautiful and sensitive score written in a Coplandesque Americana style. My surprise was great when I discovered the score to have been written by none other than John Williams himself. True he has written sensitive and poignant scores such as Schindler's List but one usually associates his name with such bombasticities as Star Wars. But in my opinion what Williams has written for this movie surpasses anything I've ever heard of his for tenderness, sensitivity and beauty, fully in keeping with the tender and lovely plot of the movie. And another recent score of his, for Catch Me if You Can, shows still more wit and sophistication. As to Stanley and Iris, I like education movies like How Green was my Valley and Konrack, that one with John Voigt and his young African American charges in South Carolina, and Danny deVito's Renaissance Man, etc. They tell a necessary story of intellectual and spiritual awakening, a story which can't be told often enough. This one is an excellent addition to that genre."
doc_features = featurize(doc, lexicon)
doc_features
```

### 2. Vectoriser un corpus

> Utiliser la fonction précédente pour vectoriser [le mini-corpus IMDB](../../data/imdb_smol.tar.gz)

Commençons par l'extraire

```bash
cd ../../local
tar -xzf ../data/imdb_smol.tar.gz 
ls -lah imdb_smol
```

Maintenant on parcourt le dossier pour construire nos représentations

```python
from collections import defaultdict
import pathlib  # Manipuler des chemins et des fichiers agréablement

def featurize_dir(corpus_root, lexicon):
    corpus_root = pathlib.Path(corpus_root)
    res = defaultdict(list)
    for clss in corpus_root.iterdir():
        # On peut aussi utiliser une compréhension de liste et avoir un dict pas default
        for doc in clss.iterdir():
            # `stem` et `read_text` c'est de la magie de `pathlib`, check it out
            res[clss.stem].append(featurize(doc.read_text(), lexicon))
    return res

# On réutilise le lexique précédent
imdb_features = featurize_dir("../../local/imdb_smol", lexicon)
imdb_features
```

## 😴 Exo 😴

### 1. Une fonction affine

> Écrire une fonction qui prend en entrée un vecteur de features et un vecteur de poids sous forme
> de tableaux numpy $x$ et $w$ de dimensions `(n,)` et un biais $b$ sous forme d'un tableau numpy de
> dimensions `(1,)` et renvoie $z=\sum_iw_ix_i + b$.

Une version élémentaire avec des boucles

```python
def affine_combination(x, w, b):
    res = np.zeros(1)
    for wi, xi in zip(w, x):
        res += wi*xi
    res += b
    return res

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

Une version plus courte avec les fonctions natives de numpy

```python
def affine_combination(x, w, b):
    return np.inner(w, x) + b

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

### 2. Un classifieur linéaire

> Écrire un classifieur linéaire qui prend en entrée des vecteurs de features à deux dimensions
> précédents et utilise les poids respectifs $0.6$ et $-0.4$ et un biais de $-0.01$. Appliquez ce
> classifieur sur le mini-corpus IMDB qu'on a vectorisé et calculez son exactitude.

On commence par définir le classifieur : on va renvoyer `True` pour la classe positive et `False`
pour la classe négative.


```python
def hardcoded_classifier(x):
    return affine_combination(x, np.array([0.6, -0.4]), -0.01) > 0.0

hardcoded_classifier(doc_features)
```

Maintenant on le teste

```python
correct_pos = sum(1 for doc in imdb_features["pos"] if hardcoded_classifier(doc))
print(f"Recall for 'pos': {correct_pos}/{len(imdb_features['pos'])}={correct_pos/len(imdb_features['pos']):.02%}")
correct_neg = sum(1 for doc in imdb_features["neg"] if not hardcoded_classifier(doc))
print(f"Recall for 'neg': {correct_neg}/{len(imdb_features['neg'])}={correct_neg/len(imdb_features['neg']):.02%}")
print(f"Accuracy: {correct_pos+correct_neg}/{len(imdb_features['pos'])+len(imdb_features['neg'])}={(correct_pos+correct_neg)/(len(imdb_features['pos'])+len(imdb_features['neg'])):.02%}")
```


On en fait une fonction, ça nous sera utile plus tard

```python
def classifier_accuracy(w, b, featurized_corpus):
    correct_pos = sum(1 for doc in featurized_corpus["pos"] if affine_combination(doc, w, b) > 0.0)
    correct_neg = sum(1 for doc in featurized_corpus["neg"] if affine_combination(doc, w, b) <= 0.0)
    return (correct_pos+correct_neg)/(len(featurized_corpus["pos"])+len(featurized_corpus["neg"]))
classifier_accuracy(np.array([0.6, -0.4]), np.array(-0.01), imdb_features)
```

## 📈 Exo 📈

> 1\. Définir la fonction `logistic` qui prend en entrée un tableau numpy $z=[z_1, …, z_n]$ et
> renvoie le tableau $[σ(z_1), … , σ(z_n)]$.

```python
def logistic(z):
    return 1/(1+np.exp(-z))
```

> 2\. Tracer avec matplotlib la courbe représentative de la fonction logistique.

```python
%matplotlib inline
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 5000)
y = logistic(x)
plt.plot(x, y)
plt.xlabel("$x$")
plt.ylabel("$σ(x)$")
plt.title("Courbe représentative de la fonction logistique sur $[-10, 10]$")
plt.show()
```


## 📉 Exo 📉

Écrire une fonction qui prend en entrée

- Un vecteur de features $x$ de taille $n$
- Un vecteur de poids $w$ de taille $n$ et un biais $b$ (de taille $1$)
- Une classe cible $y$ ($0$ ou $1$)

Et renvoie la log-vraisemblance négative du classifieur logistique de poids $(w, b)$ pour l'exemple
$(x, y)$.

Servez-vous en pour calculer le coût du classifieur de l'exercise précédent sur le mini-corpus IMDB.

### 📉 Correction 📉

```python
def logistic_negative_log_likelihood(x, w, b, y):
    g_x = logistic(affine_combination(x, w, b))
    if y == 1:
        correct_likelihood = g_x
    else:
        correct_likelihood = 1-g_x
    loss = -np.log(correct_likelihood)
    return loss
```

```python
def loss_on_imdb(w, b, featurized_corpus):
    loss_on_pos = np.zeros(1)
    for doc_features in featurized_corpus["pos"]:
        loss_on_pos += logistic_negative_log_likelihood(
            doc_features, w, b, 1
        )
    loss_on_neg = np.zeros(1)
    for doc_features in featurized_corpus["neg"]:
        loss_on_neg += logistic_negative_log_likelihood(
            doc_features, w, b, 0
        )
    return loss_on_pos + loss_on_neg
```

Avec des compréhensions

```python
def loss_on_imdb(w, b, featurized_corpus):
    loss_on_pos = sum(
        logistic_negative_log_likelihood(doc_features, w, b, 1)
        for doc_features in featurized_corpus["pos"]
    )
    loss_on_neg = sum(
        logistic_negative_log_likelihood(doc_features, w, b, 0)
        for doc_features in featurized_corpus["neg"]
    )
    return loss_on_pos + loss_on_neg
```

En version numériquement stable

```python
import math
def loss_on_imdb(w, b, featurized_corpus):
    loss_on_pos = math.fsum(
        logistic_negative_log_likelihood(doc_features, w, b, 1).astype(float)
        for doc_features in featurized_corpus["pos"]
    )
    loss_on_neg = math.fsum(
        logistic_negative_log_likelihood(doc_features, w, b, 0).astype(float)
        for doc_features in featurized_corpus["neg"]
    )
    return np.array([loss_on_pos + loss_on_neg])
```

```python
loss_on_imdb(np.array([0.6, -0.4]), -0.01, imdb_features)
```


## 🧐 Exo 🧐

### 1. Calculer le gradient

> Reprendre la fonction qui calcule la fonction de coût, et la transformer pour qu'elle renvoie le
> gradient par rapport à $w$ et la dérivée partielle par rapport à $b$ en $(x, y)$.

```python
def grad_L(x, w, b, y):
    g_x = logistic(np.inner(w, x) + b)
    grad_w = (g_x - y)*x
    grad_b = g_x - y
    return np.append(grad_w, grad_b)
grad_L(np.array([5, 10]), np.array([0.6, -0.4]), np.array([-0.01]), 0)
```

### 2. Descendre le gradient

> S'en servir pour apprendre les poids à donner aux *features* précédentes à l'aide du [mini-corpus
> IMDB](../../data/imdb_smol.tar.gz) en utilisant l'algorithme de descente de gradient stochastique.


Version minimale

```python
import random

def descent(featurized_corpus, theta_0, learning_rate, n_steps):
    train_set = [
        *((doc, 1) for doc in featurized_corpus["pos"]),
        *((doc, 0) for doc in featurized_corpus["neg"])
    ]
    theta = theta_0
    w = theta[:-1]
    b = theta[-1]
    
    for i in range(n_steps):
        # On mélange le corpus pour s'assurer de ne pas avoir d'abord tous
        # les positifs puis tous les négatifs
        random.shuffle(train_set)
        for j, (x, y) in enumerate(train_set):
            grad = grad_L(x, w, b, y)
            steepest_direction = -grad
            theta += learning_rate*steepest_direction
            w = theta[:-1]
            b = theta[-1]
    return (w, b)

descent(imdb_features, np.array([0.6, -0.4, -0.01]), 0.1, 100)
```

Avec du feedback pour voir ce qui se passe

```python
def descent_with_logging(featurized_corpus, theta_0, learning_rate, n_steps):
    train_set = [
        *((doc, 1) for doc in featurized_corpus["pos"]),
        *((doc, 0) for doc in featurized_corpus["neg"])
    ]
    theta = theta_0
    theta_history = [theta_0.tolist()]
    w = theta[:-1]
    b = theta[-1]
    print("Epoch\tLoss\tAccuracy\tw\tb")
    print(f"Initial\t{loss_on_imdb(w, b, featurized_corpus).item()}\t{classifier_accuracy(w, b, featurized_corpus)}\t{w}\t{b}")
    
    for i in range(n_steps):
        # On mélange le corpus pour s'assurer de ne pas avoir d'abord tous
        # les positifs puis tous les négatifs
        random.shuffle(train_set)
        for j, (x, y) in enumerate(train_set):
            grad = grad_L(x, w, b, y)
            steepest_direction = -grad
            # Purement pour l'affichage
            loss = logistic_negative_log_likelihood(x, w, b, y)
            #print(f"step {i*len(train_set)+j} doc={x}\tw={w}\tb={b}\tloss={loss}\tgrad={grad}")
            theta += learning_rate*steepest_direction
            w = theta[:-1]
            b = theta[-1]
        theta_history.append(theta.tolist())
        epoch_train_loss = loss_on_imdb(w, b, featurized_corpus).item()
        epoch_train_accuracy = classifier_accuracy(w, b, featurized_corpus)
        print(f"{i}\t{epoch_train_loss}\t{epoch_train_accuracy}\t{w}\t{b}")
    return (theta[:-1], theta[-1]), theta_history

theta, theta_history = descent_with_logging(imdb_features, np.array([0.6, -0.4, -0.01]), 0.1, 100)
```
