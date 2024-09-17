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

### La DOC

[À garder sous la main](https://docs.pola.rs) (ou au moins dans un onglet).

Ce notebook s'inspire du tutoriel de démarrage « [*Getting
Started*](https://docs.pola.rs/user-guide/getting-started) » de polars. C'est **vraiment important**
de prendre le temps de le lire et de le comprendre de votre côté. C'est un investissement de
quelques heures qui vous fera gagner des jours de travail. Il **faut** connaître vos outils de
travail.

## Les DataFrames

Un DataFrame c'est un tableau de données.

```python
import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "integer": [1, 2, 3],
        "date": [
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            datetime(2025, 1, 3),
        ],
        "float": [4.0, float("nan"), 5.0],
        "string": ["a", "b", "c"],
    }
)

print(df)
```

Plus précisément, un DataFrame c'est une série de **colonnes**. Chaque colonne contient une série de
données du même type.


On peut accéder à chaque colonne avec l'opérateur d'indexation de Python (les crochets droits `[]`) :

```python
df["date"]
```

Et voir le type de ses données :

```python
df["date"].dtype
```

Attention : le type **contenu** par une colonne c'est son `dtype`. Voici son `type` au sens de Python

```python
type(df["date"])
```

Rappel

```python
type([1, 2, 3, 4])
```

Les colonnes ne sont donc pas des simples listes mais des objets qui, comme les DataFrames, sont
spécifiques à Polars, avec des interfaces pour s'en servir aisément.


En plus d'un ensemble de colonnes, un DataFrame peut aussi être vu comme un ensemble de **lignes de données** (*rows*), représentées par défaut comme des `tuples` de données.

```python
df.row(1)
```

```python
for r in df.iter_rows():
    print(r)
```

Mais il est souvent plus pratique (même si c'est un peu plus lent) de les lire comme des dicts :

```python
df.row(1, named=True)
```

```python
for r in df.iter_rows(named=True):
    print(r)
```

Attention, contrairement à Pandas, les Dataframes de Polars sont **immutables**, ce qui signifie que ceci ne modifie rien :

```python
df.row(1, named=True)["string"] = "truc"
df
```

Ça peut paraître limitant, mais c'est en fait une bonne chose : Polars a des fonctions bien plus efficaces pour faire ce dont vous avez besoin (promis).


Enfin, polars permet de lire et d'écrire dans la plupart des formats de stockage de données standard, dont le très lisible `csv` :

```python
df.write_csv("monfichier.csv")
df_csv = pl.read_csv("monfichier.csv")
print(df_csv)
```

## Opérations

Le principal intérêt de Polars c'est de permettre de décrire très facilement des opérations sur les DataFrames et de les exécuter efficacement.

### Sélections


`select` permet de sélectionner des colonnes.

```python
df.select(pl.col("integer", "string"))
```

Et `filter` de sélectionner des **lignes**

```python
df.filter(
    pl.col("date").is_between(datetime(2025, 12, 2), datetime(2025, 12, 3)),
)
```

```python
df.filter((pl.col("integer") < 3) & (pl.col("float").is_not_nan()))
```

### Mutations

`with_columns` permet d'ajouter des colonnes. En général on fait ça pour y stocker le résultat d'une opération :

```python
df.with_columns((pl.col("integer") + 2713).alias("mon résultat"))
```

Mais ! On avait pas dit que les DataFrames étaient immutables ?????


Pas de panique :

```python
df
```

La méthode `with_columns` **renvoie** un nouveau DataFrame, elle ne modifie pas directement. Ça peut sembler couteux en mémoire, mais tout est fait pour que ça ne le soit pas. En particulier les colonnes communes à ces deux DataFrames partagent le même espace mémoire (ce qui serait un problème si elles étaient mutables, mais précisément, elles ne le sont pas !).


Si on veut stocker ce nouveau DataFrame, il faut le faire explicitement

```python
df2 = df.with_columns((pl.col("integer") + 2713).alias("mon résultat"))
print(df2)
```

Pour être plus efficace on peut aussi construire plusieurs colonnes d'un coup :

```python
df.with_columns(
    (pl.col("integer") + 2713).alias("mon résultat"),
    pl.col("float").add(pl.col("integer")).alias("e"),
)
```

et enchaîner les transformations

```python
df.with_columns(
    pl.col("float").add(pl.col("integer")).alias("e"),
).with_columns(
    pl.col("e").add(1).alias("f"),
)
```

Cette dernière façon d'écrire est très utile pour décrire des séries de traitements de données et je vous encourage très fort à l'utiliser. En plus, si Polars est en mode *lazy* (allez voir [la doc](https://docs.pola.rs/user-guide/concepts/lazy-vs-eager/), etc.), il optimisera automatiquement les opérations que vous demandez, ce qui peut être un gain de temps énorme quand il y a beaucoup de données à traiter.


Il y a vraiment beaucoup d'opérations qui existent déjà dans Polars et sont optimisées et il faut toujours commencer par là et chercher attentivement, mais des fois il n'y pas le truc exact dont on a besoin. Il y a alors deux options :


Soit ce dont vous avez besoin existe dans Numpy (ou vous êtes capable de créer une `ufunc` vous mêmes). Dans ce cas c'est utilisable directement :

```python
import numpy as np

df.with_columns(np.sqrt(pl.col("integer")).alias("sqrt"))
```

Soit vraiment vous avez besoin de plus de personnalisation :

```python
def ma_fonction_tordue(s):
    if s == "a":
        return 2713
    if s == "b":
        return -2
    return 0
```

Dans ce cas on peut appliquer une fonction qu'on aura défini en Python. Attention c'est **beaucoup** plus lent :

```python
df.with_columns(pl.col("string").map_elements(ma_fonction_tordue).alias("bidule"))
```

Dans ce cas c'est mieux de préciser le type de données renvoyé

```python
df.with_columns(
    pl.col("string").map_elements(
        ma_fonction_tordue, return_dtype=str
    ).alias("bidule")
)
```
