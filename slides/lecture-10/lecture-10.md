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
Cours 10 : Régression logistique
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-11
<!-- #endregion -->

```python
from IPython.display import display
```

## Classifieur logistique

## 🧠 Exo 🧠

1\. Tracer avec matplotlib la courbe représentative de la fonction logistique.

2\. À l'aide d'un lexique de sentiment (par exemple [VADER](https://github.com/cjhutto/vaderSentiment)), écrivez un classifieur logistique à deux features : nombre
de mots positifs et nombre de mots négatifs avec les poids respectifs $0.6$ et $0.4$ et pas de terme
de biais. Appliquez ce classifieur sur le mini-corpus LMDB et calculez son exactitude.

## Apprendre un classifieur logistique

## 😩 Exo 😩

1\. Coder la log-vraisemblance négative et son gradient

2\. S'en servir pour apprendre les poids à donner aux features précédentes à l'aide du mini-corpus LMDB
