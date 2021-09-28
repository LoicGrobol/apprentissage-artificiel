---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[comment]: <> "LTeX: language=fr"

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
np.zeros(4)
```

- np.ones

```python
np.ones(3)
```

- np.arange

```python
np.arange(10)
```

- np.linspace(start, stop) (crée un *array* avec des valeurs réparties uniformément entre start et
   stop (50 valeurs par défaut))

```python
np.linspace(0, 10, num=5)
```

- np.empty (crée un array vide, enfin avec des valeurs aléatoires)

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
b[1,:] # 2e ligne, toutes les colonnes
```

```python
b[:,3] # 4e colonne, toutes les lignes
```

- On peut aussi faire des sélections avec des conditions (oui comme dans pandas)

```python
a[a > 2]
```

```python
a[a%2 == 0]
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

## Opérations

- Les trucs classiques

```python
a.sum()
```

```python
a.max()
```

```python
a.min()
```

```python
c = np.arange(10,16)
```

- Opérations sur *array* à une dimension

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
a / c
```

- Poduit matriciel

```python
m1 = np.array([[1, 2],[ 3, 4]])
```

```python
m2 = np.array([[5, 6],[ 7, 8]])
```

```python
m1@m2
```

```python
np.dot(m1, m2)
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
plt.plot(np.random.random(20))
```

Après dès qu'on veut faire des trucs un peu plus compliqué ben ça devient plus compliqué matplotlib.

Mais on peut aussi faire des trucs fun assez facilement. Exemple avec une image.  
`plt.imread` permet de changer un fichier image en objet python… devinez lequel

```python
im = plt.imread("data/the-queens-gambit.jpeg")
type(im)
```

Bingo, un *array* numpy. En même temps c'est jamais qu'une matrice de pixels une image. 

```python
im.shape
```

Un *array* à trois dimensions : X, Y (les coordonnées du pixel) et la valeur RGB du pixel

Le pixel `(200, 200)` par exemple est un *array* de 3 éléments `(r,g,b)` :

```python
# 
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
