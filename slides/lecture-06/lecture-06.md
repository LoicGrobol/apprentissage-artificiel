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
Cours 6 : NumpPy
=================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-29
<!-- #endregion -->

```python
from IPython.display import display
```

## NumPy ?

[NumPy](https://numpy.org/).

NumPy est un des packages les plus utilisés de Python. Il ajoute au langage le
support des tableaux multidimensionnels (`ndarray`) et du calcul matriciel.

Installons NumPy, soit dans votre terminal avec `pip`, soit en exécutant la cellule de code
suivante. Comme d'habitude, il est vivement recommandé de travailler pour ce cours dans un
[environnement virtuel](../lecture-05/lecture-05.md) et si vous avez installé le
[requirements.txt](../../requirements.txt) de ce cours, NumPy est déjà installé.


```python
%pip install -U numpy
```

```python
import numpy as np
```

Ne faites pas autrement, c'est devenu une formule consacrée. Faire autrement, c'est de la perversion

![](https://i.redd.it/eam52i3vyny41.jpg)

## `ndarray`

Le grand apport de NumPy ce sont les *array* (classe `ndarray`), à une dimension (vecteur), deux
dimensions (matrices) ou trois et plus (tenseur).

Un *array* sera plus rapide et plus compact (moins de taille en mémoire) qu'une liste Python.

NumPy ajoute plein de fonctions pour manipuler ses *array* de façon optimisée. À tel point qu'il est
recommandé de ne pas utiliser de boucle pour les manipuler.

On peut créer un *array* à partir d'une liste (ou d'un tuple) :

```python
a = np.array([1, 2, 3, 4, 5, 6]) # une dimension
```

ou d'une liste de listes

```python
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) #deux dimensions
```

MAIS à la différence d'une liste, un *array* aura les caractéristiques suivantes :

- Une taille fixe (donnée à la création)
- Ses éléments doivent tous être de même type

### Infos sur les `ndarray`

Pour avoir des infos sur les *array* que vous manipulez vous avez :

- `dtype` (type des éléments)

```python
b.dtype
```

- `ndim` (le nombre de dimensions)

```python
print(a.ndim)
print(b.ndim)
```

- `size` (le nombre d'éléments)

```python
b.size
```

- `shape` (un tuple avec la taille de chaque dimension)

```python
b.shape
```

### Créer un *array*

- `np.zeros`

```python
display(np.zeros(4))
display(np.zeros((3,4)))
display(np.zeros((3,4,5)))
display(np.zeros((3,4), dtype=int))
```

- `np.ones`

```python
np.ones(3)
```

```python
np.full((3,4), fill_value=2)
```

```python
np.eye(4)
```

- `np.arange`

```python
np.arange(10)
```

- `np.linspace(start, stop)` (crée un *array* avec des valeurs réparties uniformément entre start et
   stop (50 valeurs par défaut))

```python
np.linspace(0, 10, num=5)
```

- `np.empty` (crée un array vide, enfin avec des valeurs aléatoires)

```python
np.empty(8)
```

### Indexer et trancher

- Comme pour les listes Python

```python
a[4]
```

```python
a[:2]
```

- Au-delà d'une dimension il y a une syntaxe différente

```python
b
```

```python
b[1,1] # 2e ligne, 2e colonne
```

```python
display(b[1,:]) # 2e ligne, toutes les colonnes
display(b[1])
display(b[1][:2])
```

```python
b[:,3] # 4e colonne, toutes les lignes
```

- On peut aussi faire des sélections avec des conditions (oui comme dans pandas)

```python
a
```

```python
a[a > 2]
```

```python
a > 2
```

```python
a[a%2 == 0]
```

```python
a[a%2]
```

```python
display(a%2 == 0)
display(a%2)
```

## Changer de dimension

```python
c = np.arange(6)
print(c)
```

J'en fais une matrice de 2 lignes et 3 colonnes

```python
c.reshape(2, 3)
```

On revient à une dimension

```python
print(c.flatten())
```

Hop, on ajoute une dimension

```python
c[:,np.newaxis]
```

Transposition (lignes deviennent colonnes et colonnes deviennent lignes)

```python
c2 = c.reshape(2, 3)
print(c2)
c2.T
```

```python
c2.T.flatten()
```

## Opérations

- Les trucs classiques

```python
a
```

```python
a.sum()
```

```python
a.max()
```

```python
a.argmax()
```

```python
a.min()
```

```python
c = np.arange(10,16)
c
```

- Opérations sur *array* à une dimension

```python
a = np.arange(6)
```

```python
a + c
```

```python
a - c
```

```python
a * c
```

```python
a/c
```

- Produit matriciel

```python
m1 = np.array([[1, 2],[ 3, 4]])
m1
```

```python
m2 = np.array([[5, 6],[ 7, 8]])
m2
```

```python
m1@m2
```

```python
np.matmul(m1, m2)
```

## Broadcasting

Une notion un peu plus compliquée mais qui sert souvent

```python
a = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
a
```

```python
c = np.array([2, 4, 8])
c
```

```python
a+c
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Explication : si un des tableaux a moins de dimensions que l'autre, numpy fait automatiquement la conversion pour que tout se passe comme si on avait ajouté par
<!-- #endregion -->

```python
np.broadcast_to(c, [3,3])
```

Ajouter un tableau à une dimension revient donc à ajouter colonne par colonne

```python
a*-1
```

## Matplotlib

Les deux packages sont très copains, c'est très simple d'afficher des graphiques à partir de données
NumPy. Installez-le d'abord si c'est nécessaire, vous savez faire, maintenant.

```python
import matplotlib.pyplot as plt
```

```python
%matplotlib inline
plt.plot(a)
```

```python
a = np.random.random(20)
display(a)
plt.plot(a)
```

Après dès qu'on veut faire des trucs un peu plus compliqué ben ça devient plus compliqué matplotlib.

Mais on peut aussi faire des trucs fun assez facilement. Exemple avec une image.  
`plt.imread` permet de changer un fichier image en objet python… devinez lequel

```python
im = plt.imread("../../data/the-queens-gambit.jpeg")
type(im)
```

Bingo, un *array* numpy. En même temps c'est jamais qu'une matrice de pixels une image. 

```python
im.shape
```

Un *array* à trois dimensions : X, Y (les coordonnées du pixel) et la valeur RGB du pixel

Le pixel `(200, 200)` par exemple est un *array* de 3 éléments `(r,g,b)` :

```python
im[200,200]
```

Oui on peut voir l'image aussi

```python
plt.imshow(im) 
```

si je ne prends que la valeur de R dans RGB j'obtiens des niveaux de gris (ça marche aussi pour G ou
B)

```python
plt.imshow(im[:,:,0])
```

Magie

```python
plt.imshow(im[:,:,0], cmap=plt.get_cmap('gray'))
```

Si vous voulez en savoir plus je vous invite à consulter les pages suivantes :

- <https://matplotlib.org/tutorials/introductory/images.html>
- <https://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/>

## S'entraîner avec NumPy

Pour vous entraîner à manipuler des *arrays* et découvrir les fonctions de NumPy. Je vous recommande
la série d'exercices corrigés à <https://www.w3resource.com/python-exercises/numpy/index-array.php>.
Essayez au maximum de les résoudre sans écrire de boucles.

## 👜 Exo : les sacs de mots 👜

### 1. Faire des sacs

- Écrire un script qui prend en entrée un dossier contenant des documents (sous forme de fichier
  textes) et sort un fichier TSV donnant pour chaque document sa représentation en sac de mots (en
  nombre d'occurrences des mots du vocabulaire commun)
  - Dans le sens habituel : un fichier par ligne, un mot par colonne
  - Pour itérer sur les fichiers dans un dossier on peut utiliser `for f in
    pathlib.Path(chemin_du_dossier).glob('*')`
  - Pour récupérer des arguments en ligne de commande :
    [`argparse`](https://docs.python.org/3/library/argparse.html) ou
    [`sys.argv`](https://docs.python.org/3/library/argparse.html)
- Tester sur la partie positive du [mini-corpus imdb](../../data/imdb_smol.tar.gz)

Pensez à ce qu'on a vu les cours précédents pour ne pas réinventer la roue.

### 2. Faire des sacs relatifs

Modifier le script précédent pour qu'il génère des sacs de mots utilisant les fréquences relatives
plutôt que les nombres d'occurrences.

### 3. Faire des tfidsacs


Modifier le script de précédent pour qu'il renvoie non plus les fréquences relatives de chaque mot
mais leur tf⋅idf avec la définition suivante pour un mot $w$, un document $D$ et un corpus $C$

- $\mathrm{tf}(w, D)$ est la fréquence relative de $w$ dans $D$
- $$\mathrm{idf}(w, C) = \log\!\left(\frac{\text{nombre de documents dans $C$}}{\text{nombre de
  documents de $C$ qui contiennent $w$}}\right)$$
- $\log$ est le logarithme naturel
  [`np.log`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)
- $\mathrm{tfidf}(w, D, C) = \mathrm{tf}(w, D)×\mathrm{idf}(w, C)$

Pistes de recherche :

- L'option `keepdims` de `np.sum`
- `np.transpose`
- `np.count_nonzero`
- Regarder ce que donne `np.array([[1, 0], [2, 0]]) > 0`
