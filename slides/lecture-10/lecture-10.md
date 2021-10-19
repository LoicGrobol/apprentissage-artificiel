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

2021-10-11
<!-- #endregion -->

```python
from IPython.display import display
```

## Classifieur linéaire

## 😴 Exo 😴

Écrire une fonction qui prend en entrée un vecteur de features et un vecteur de poids sous forme de
tableaux numpy $x$ et $w$ de dimensions `(n,)` et un biais $b$ sous forme d'un tableau numpy de
dimensions `(1,)` et renvoie $z=\sum_iw_ix_i + b$.

## La fonction logistique

## 📈 Exo 📈

Tracer avec matplotlib la courbe représentative de la fonction logistique.

## Classifieur logistique

## 🧠 Exo 🧠

1\. À l'aide d'un lexique de sentiment (par exemple
[VADER](https://github.com/cjhutto/vaderSentiment)), écrivez une fonction qui prend en entrée un
texte en anglais et renvoie sa représentation sous forme d'un vecteur de features à deux traits :
nombre de mots positifs et nombre de mot négatifs.

2\. Appliquer la fonction précédente sur le mini-corpus IMDB

3\. Écrire un classifieur logistique (en une fonction) qui prend en entrée les vecteurs de features
précédents et utilise les poids respectifs $0.6$ et $0.4$ et pas de terme de biais. Appliquez ce
classifieur sur le mini-corpus IMDB et calculez son exactitude.

## Fonction de coût

## 📉 Exo 📉

Écrire une fonction qui prend en entrée

- Un vecteur de features $x$ de taille $n$
- Un vecteur de poids $w$ de taille $n$ et un biais $b$ (de taille $1$)
- Une classe cible $c$ ($0$ ou $1$)

Et renvoie la log-vraisemblance négative du classifieur logistique de poids $(w, b)$ pour l'exemple
$(x, c)$.

## Descente de gradient

## 🧐 Exo 🧐

Reprendre la fonction qui calcule la fonction de coût, mais faire en sorte qu'elle renvoie également
le gradient en $(x, c)$.

## 😩 Exo 😩

S'en servir pour apprendre les poids à donner aux features précédentes à l'aide du mini-corpus LMDB
