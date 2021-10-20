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
Cours 10 : Régression logistique
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-20
<!-- #endregion -->

```python
from IPython.display import display
```

## Classifieur linéaire

On considère des vecteurs de *features* de dimension $n$

$$\mathbf{x} = (x₁, …, x_n)$$

Un vecteur de poids de dimension $n$

$$\mathbf{w} = (w₁, …, w_n)$$

et un biais $b$ scalaire (un nombre quoi).

Pour réaliser une classification on considère le nombre $z$ (on parle parfois de *logit*)

$$z=w₁×x₁ + … + w_n×x_n + b = \sum_iw_ix_i + b$$

Ce qu'on note aussi

$$z = \mathbf{w}⋅\mathbf{x}+b$$

$\mathbf{w}⋅\mathbf{x}$ se lit « w scalaire x », on parle de *produit scalaire* en français et de *inner product* en anglais.

(ou pour les mathématicien⋅ne⋅s acharné⋅e⋅s $z = \langle w\ |\ x \rangle + b$)

Quelle que soit la façon dont on le note, on affectera à $\mathbf{x}$ la classe $0$ si $z < 0$ et la classe $1$ sinon.

## 😴 Exo 😴

Écrire une fonction qui prend en entrée un vecteur de features et un vecteur de poids sous forme de
tableaux numpy $x$ et $w$ de dimensions `(n,)` et un biais $b$ sous forme d'un tableau numpy de
dimensions `(1,)` et renvoie $z=\sum_iw_ix_i + b$.

```python
import numpy as np
```

```python
def affine_combination(x, w, b):
    pass # À vous de jouer !

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

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

```python
def affine_combination(x, w, b):
    return np.inner(w, x) + b

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

## La fonction logistique


$$σ(z) = \frac{1}{1 + e^{−z}} = \frac{1}{1 + \exp(−z)}$$

Elle permet de *normaliser* $z$ : $z$ peut être n'importe quel nombre entre $-∞$ et $+∞$, mais on aura toujours $0 < σ(z) < 1$, ce qui permet de l'interpréter facilement comme une *vraisemblance*. Autrement dit, $σ(z)$ sera proche de $1$ s'il paraît vraisemblable que $x$ appartienne à la classe $1$ et proche de $0$ sinon.

## 📈 Exo 📈

Tracer avec matplotlib la courbe représentative de la fonction logistique.

```python
def logistic(z):
    return 1/(1+np.exp(-z))
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 5000)
y = logistic(x)
plt.plot(x, y)
```

## Régression logistique


Formellement : on suppose qu'il existe une fonction $f$ qui prédit parfaitement les classes, donc
telle que pour tout couple exemple/étiquette $(x, y)$ avec $y$ valant $0$ ou $1$, $f(x) = y$. On
approcher cette fonction par une fonction $g$ de la forme

$$g(x) = σ(w⋅x+b)$$

Si on choisit les poids $w$ et le biais $b$ tels que $g$ soit la plus proche possible de $f$ sur
notre ensemble d'apprentissage, on dit que $g$ est la *régression logistique de $f$* sur cet
ensemble.

Un classifieur logistique, c'est simplement un classifieur qui pour un exemple $x$ renvoie $0$ si
$g(x) < 0.5$ et $1$ sinon.

## 🧠 Exo 🧠

1\. À l'aide d'un lexique de sentiment (par exemple
[VADER](https://github.com/cjhutto/vaderSentiment)), écrivez une fonction qui prend en entrée un
texte en anglais et renvoie sa représentation sous forme d'un vecteur de features à deux traits :
nombre de mots positifs et nombre de mot négatifs.

```python
import re

def poor_mans_tokenizer_and_normalizer(s):
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]

def read_vader(vader_path):
    res = dict()
    with open(vader_path) as in_stream:
        for row in in_stream:
            word, polarity, *_ = row.lstrip().split("\t", maxsplit=2)
            is_positive = float(polarity) > 0
            res[word] = is_positive
    return res

def featurize(text, lexicon):
    words = poor_mans_tokenizer_and_normalizer(text)
    features = np.empty(2)
    features[0] = sum(1 for w in words if lexicon.get(w, False))
    features[1] = sum(1 for w in words if not lexicon.get(w, True))
    return features

lexicon = read_vader("../../data/vader_lexicon.txt")
doc = "I came in in the middle of this film so I had no idea about any credits or even its title till I looked it up here, where I see that it has received a mixed reception by your commentators. I'm on the positive side regarding this film but one thing really caught my attention as I watched: the beautiful and sensitive score written in a Coplandesque Americana style. My surprise was great when I discovered the score to have been written by none other than John Williams himself. True he has written sensitive and poignant scores such as Schindler's List but one usually associates his name with such bombasticities as Star Wars. But in my opinion what Williams has written for this movie surpasses anything I've ever heard of his for tenderness, sensitivity and beauty, fully in keeping with the tender and lovely plot of the movie. And another recent score of his, for Catch Me if You Can, shows still more wit and sophistication. As to Stanley and Iris, I like education movies like How Green was my Valley and Konrack, that one with John Voigt and his young African American charges in South Carolina, and Danny deVito's Renaissance Man, etc. They tell a necessary story of intellectual and spiritual awakening, a story which can't be told often enough. This one is an excellent addition to that genre."
doc_features = featurize(doc, lexicon)
doc_features
```

2\. Appliquer la fonction précédente sur [le mini-corpus IMDB](../../data/imdb_smol.tar.gz)


3\. Écrire un classifieur logistique qui prend en entrée les vecteurs de features
précédents et utilise les poids respectifs $0.6$ et $-0.4$ et un biais de $0$. Appliquez ce
classifieur sur le mini-corpus IMDB et calculez son exactitude.

```python
def hardcoded_classifier(x):
    return logistic(
        np.inner(
            np.array([0.6, -0.4]),
            x,
        )
    )

hardcoded_classifier(doc_features)
```

## Fonction de coût

On formalise « être le plus proche possible » de la section précédente comme minimiser une certaine
fonction de coût (*loss*) $L$.

Autrement dit, étant donné un ensemble de test $(x₁, y₁), …, (x_n, y_n)$, on va mesurer la qualité
du classifieur logistique $g$

$$\mathcal{L} = \sum_i L(g(xᵢ), yᵢ)$$

Dans le cas de la régression logistique, on va utiliser la *log-vraisemblance négative* (*negative
log-likelihood*) :

On définit la *vraisemblance* $V$ comme
$$
V(a, y) =
    \begin{cases}
        a & \text{si $y = 1$}\\
        1-a & \text{sinon}
    \end{cases}
$$

Intuitivement, il s'agit de la vraisemblance affectée par le modèle à la classe correcte $y$. Il ne
s'agit donc pas d'un coût, mais d'un *gain* (si sa valeur est haute, c'est que le modèle est bon)

La *log-vraisemblance négative* $L$ est alors définie par

$$L(a, y) = -\log(V(a, y))$$

Le $\log$ permet de régulariser la valeur (c'est plus facile à apprendre) et le $-$ à s'assurer
qu'on a bien un coût (plus la valeur est basse, meilleur le modèle est).

Une autre façon de le voir : $L(a, y)$, c'est la
[surprise](https://en.wikipedia.org/wiki/Information_content) de $y$ au sens de la théorie de
l'information. Autrement dit : si j'estime qu'il y a une probabilité $a$ d'observer la classe $y$,
$L(a, y)$ mesure à quel point il serait surprenant d'observer effectivement $y$.


On peut l'écrire en une ligne : pour un exemple $x$, le coût de l'exemple $(x, y)$ est

$$L(g(x), y) = -\log\left[g(x)×y + (1-g(x))×(1-y)\right]$$

C'est un *trick*, l'astuce c'est que comme $y$ vaut soit $0$ soit $1$, soit $y=0$, soit $1-y=0$ et
donc la somme dans le $\log$ se simplifie dans tous les cas. Rien de transcendant là-dedans.

La formule diffère un peu de celle de *Speech and Language Processing* mais les résultats sont les
mêmes et celle-ci est mieux pour notre problème !

<small>En fait la leur est la formule générale de l'entropie croisée pour des distributions de proba
à support dans $\{0, 1\}$, ce qui est une autre intuition pour cette fonction de coût, mais ici elle
nous complique la vie.</small>

## 📉 Exo 📉

Écrire une fonction qui prend en entrée

- Un vecteur de features $x$ de taille $n$
- Un vecteur de poids $w$ de taille $n$ et un biais $b$ (de taille $1$)
- Une classe cible $y$ ($0$ ou $1$)

Et renvoie la log-vraisemblance négative du classifieur logistique de poids $(w, b)$ pour l'exemple
$(x, y)$.

Servez vous-en pour calculer le coût du classifieur de l'exercise précédent sur le mini-corpus IMDB.

## Descente de gradient

Pour un classifieur logistique

$$\frac{∂L(g(x, y))}{∂w_i} = (g(x)-y)×x_i$$

et

$$\frac{∂L(g(x, y))}{∂b} = (g(x)-y)$$

## 🧐 Exo 🧐

Reprendre la fonction qui calcule la fonction de coût, mais faire en sorte qu'elle renvoie également
le gradient par rapport à $w$ et la dérivée partielle par rapport à $b$ en $(x, y)$.

## 😩 Exo 😩

S'en servir pour apprendre les poids à donner aux features précédentes à l'aide du  [mini-corpus
IMDB](../../data/imdb_smol.tar.gz) en utilisant l'algorithme de descente de gradient stochastique.
