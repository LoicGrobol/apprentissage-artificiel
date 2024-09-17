---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cours-ml
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
<!-- LTeX: language=fr -->


TP 3 : Polars
=================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Polars ?

[Polars](https://pola.rs/).

Polars est une bibliothèque de gestion de données tabulaires (les `DataFrames` que vous avez déjà
rencontré dans `pandas` et/ou dans R et dont [l'origine est le langage
S](https://towardsdatascience.com/preventing-the-death-of-the-dataframe-8bca1c0f83c8)). Elle a
l'avantage sur `pandas` d'être **beaucoup** plus rapide, efficace et ergonomique.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Installation
<!-- #endregion -->

Comme d'habitude :

- Avoir un environnement virtuel.
- `python -m pip install polars` dans un terminal où l'environnement virtuel a été activé.
- ???
- profit

