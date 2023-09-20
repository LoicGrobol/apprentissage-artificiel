---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
<!-- LTeX: language=fr -->


Cours 5 : NumPy
=================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-10-12
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## NumPy ?

[NumPy](https://numpy.org/).

NumPy est un des packages les plus utilisés de Python. Il ajoute au langage des maths plus performantes, le support des tableaux multidimensionnels (`ndarray`) et du calcul matriciel.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Installation

[Comme on l'a dit](../04-pip_venv/pip-venv.py.md), il est vivement recommandé de travailler dans un environnement virtuel.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Si vous avez installé le
[requirements.txt](../../requirements.txt) de ce cours, NumPy est déjà installé.

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Sinon installez NumPy, soit dans votre terminal avec `pip`, soit en exécutant la cellule de code
suivante :
<!-- #endregion -->

```python
%pip install -U numpy
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On importe Numpy comme ceci
<!-- #endregion -->

```python
import numpy as np
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Ne faites pas autrement, c'est devenu une formule consacrée. Faire autrement, notamment en lui
donnant un autre nom, c'est de la perversion
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
![](https://i.redd.it/eam52i3vyny41.jpg)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Maths de base
<!-- #endregion -->

Numpy a ses propres types numériques, qui permettent par exemple de travailler avec différentes précisions.

```python slideshow={"slide_type": "-"}
# Un nombre à virgule flottante codé sur 16 bits
half = np.float16(1.0)
type(half)
```

```python
# Un nombre à virgule flottante codé sur 32 bits
single = np.float32(1.0)
type(single)
```

```python
# Un nombre à virgule flottante codé sur 64 bits
double = np.float64(1.0)
type(double)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut faire des maths comme d'habitude avec
<!-- #endregion -->

```python
print(double + double, type(double + double))
```

```python
print(double + single, type(double + single))
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Et même les combiner avec les types habituels de Python
<!-- #endregion -->

```python
print(double + 1.0, type(double + 1.0))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Et Numpy vous donne accès à plein de fonctions mathématiques, souvent plus efficaces que les équivalents du module standard `math`, plus variées, et apportant souvent d'autres avantages, comme une meilleur stabilité numérique.
<!-- #endregion -->

```python
np.log(1.5)
```

```python
np.logaddexp(2.7, 1.3)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## `ndarray`

Le grand apport de NumPy ce sont les *array* (classe `ndarray`), à une dimension (vecteur), deux
dimensions (matrices) ou trois et plus (tenseur).

Un *array* sera plus rapide et plus compact (moins de taille en mémoire) qu'une liste Python.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
NumPy ajoute plein de fonctions pour manipuler ses *array* de façon optimisée. À tel point qu'il est
recommandé de ne pas utiliser de boucle pour les manipuler.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut créer un *array* à partir d'une liste (ou d'un tuple) :
<!-- #endregion -->

```python
a = np.array([1, 2, 3, 4, 5, 6]) # une dimension
a
```

ou d'une liste de listes

```python
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) #deux dimensions
b
```

<!-- #region slideshow={"slide_type": "subslide"} -->
**Mais** à la différence d'une liste, un *array* aura les caractéristiques suivantes :

- Une taille fixe (donnée à la création)
- Ses éléments doivent tous être de même type
<!-- #endregion -->

```python tags=["raises-exception"] slideshow={"slide_type": "fragment"}
b.append(1)
```

```python slideshow={"slide_type": "subslide"}
a = np.array([1, 1.2])
a
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Infos sur les `ndarray`

Pour avoir des infos sur les *array* que vous manipulez vous avez :

- `dtype` (type des éléments)
<!-- #endregion -->

```python
b.dtype
```

- `ndim` (le nombre de dimensions)

```python
print(a.ndim)
print(b.ndim)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `size` (le nombre d'éléments)
<!-- #endregion -->

```python
b.size
```

- `shape` (un tuple avec la taille de chaque dimension)

```python
b.shape
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Créer un *array*

- `np.zeros`
<!-- #endregion -->

```python
np.zeros(4)
```

```python slideshow={"slide_type": "fragment"}
np.zeros((3,4))
```

```python slideshow={"slide_type": "subslide"}
np.zeros((2,3,4))
```

```python slideshow={"slide_type": "fragment"}
np.zeros((3,4), dtype=int)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `np.ones`
<!-- #endregion -->

```python
np.ones(3)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
- `np.full`
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
np.full((3,4), fill_value=2)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Et des choses plus sophistiquées
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
- `np.eye`
<!-- #endregion -->

```python
np.eye(4)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
- `np.arange`
<!-- #endregion -->

```python
np.arange(10)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `np.linspace(start, stop)` (crée un *array* avec des valeurs réparties uniformément entre start et
   stop (50 valeurs par défaut))
<!-- #endregion -->

```python
np.linspace(0, 10, num=5)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
- `np.empty` (crée un array vide, enfin avec des valeurs aléatoires)
<!-- #endregion -->

```python
np.empty(8)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Il y a plein d'autres !
<!-- #endregion -->

```python
np.random.rand(3,2)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Allez lire [la doc](https://numpy.org/doc) 👀
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Indexer et trancher

- Comme pour les listes Python
<!-- #endregion -->

```python
a = np.array([5, 6, 7, 8, 9, 10])
a
```

```python slideshow={"slide_type": "fragment"}
a[4]
```

```python slideshow={"slide_type": "fragment"}
a[:2]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- Au-delà d'une dimension il y a une syntaxe différente
<!-- #endregion -->

```python
b = np.random.randint(13, 27, size=(5, 7))
b
```

```python slideshow={"slide_type": "fragment"}
b[1,2] 
```

```python slideshow={"slide_type": "fragment"}
b[1,:] # 2e ligne, toutes les colonnes
```

```python slideshow={"slide_type": "subslide"}
b[:,3] # 4e colonne, toutes les lignes, attention à la dimension !
```

```python slideshow={"slide_type": "subslide"}
b[1][:2]
```

```python slideshow={"slide_type": "fragment"}
b[1]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- On peut aussi faire des sélections avec des conditions (oui comme dans pandas)
<!-- #endregion -->

```python
a = np.array([5, 6, 7, 8, 9, 10])
a
```

```python slideshow={"slide_type": "fragment"}
a[a > 7]
```

```python slideshow={"slide_type": "fragment"}
a > 7
```

```python slideshow={"slide_type": "subslide"}
a[a%2 == 0]
```

```python slideshow={"slide_type": "fragment"}
a[a%2 == 1]
```

```python
a%2 == 0
```

```python
a%2 == 1
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Changer de dimension
<!-- #endregion -->

```python
c = np.arange(6)
c
```

<!-- #region slideshow={"slide_type": "fragment"} -->
J'en fais une matrice de 2 lignes et 3 colonnes
<!-- #endregion -->

```python
c.reshape(2, 3)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On revient à une dimension
<!-- #endregion -->

```python
c.flatten()
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Hop, on ajoute une dimension
<!-- #endregion -->

```python
c[:, np.newaxis]
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Transposition (lignes deviennent colonnes et colonnes deviennent lignes)
<!-- #endregion -->

```python
c2 = c.reshape(2, 3)
print(c2)
print(c2.T)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Opérations

- Les trucs classiques
<!-- #endregion -->

```python
a = np.array([5, 6, 7, 8, 9, 10])
a
```

```python
a.sum()
```

```python slideshow={"slide_type": "subslide"}
a.max()
```

```python
a.argmax()
```

```python
a.min()
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- Opérations sur *array* à une dimension
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
c = np.arange(10,16)
c
```

```python
a = np.arange(6)
a
```

```python slideshow={"slide_type": "subslide"}
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

<!-- #region slideshow={"slide_type": "subslide"} -->
- Produit matriciel
<!-- #endregion -->

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

<!-- #region slideshow={"slide_type": "slide"} -->
## Broadcasting

Une notion un peu plus compliquée, mais qui sert souvent.
<!-- #endregion -->

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
Explication : si un des tableaux a moins de dimensions que l'autre, Numpy fait automatiquement la
conversion pour que tout se passe comme si on avait ajouté par
<!-- #endregion -->

```python
np.broadcast_to(c, [3,3])
```

Ajouter un tableau à une dimension revient donc à ajouter colonne par colonne

```python
a*-1
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Pensez à ?
<!-- #endregion -->

À [lire la doc](https://numpy.org/doc/stable/user/basics.broadcasting.html).


<!-- #region slideshow={"slide_type": "slide"} -->
## Matplotlib

Les deux packages sont très copains, c'est très simple d'afficher des graphiques à partir de données
NumPy. Installez-le d'abord si c'est nécessaire, vous savez faire, maintenant.
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
```

```python
%matplotlib inline
plt.plot(np.array([1, 2, 4, 8, 16]))
```

```python slideshow={"slide_type": "subslide"}
a = np.random.random(20)
display(a)
plt.plot(a)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Après dès qu'on veut faire des trucs un peu plus compliqués ben ça devient plus compliqué,
matplotlib.

Mais on peut aussi faire des trucs fun assez facilement. Exemple avec une image.
<!-- #endregion -->

`plt.imread` permet de changer un fichier image en objet python… devinez lequel

```python
im = plt.imread("../../data/the-queens-gambit.jpeg")
type(im)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Bingo, un *array* Numpy. En même temps, c'est jamais qu'une matrice de pixels une image.
<!-- #endregion -->

```python
im.shape
```

Un *array* à trois dimensions : X, Y (les coordonnées du pixel) et la valeur RGB du pixel

Le pixel `(200, 200)` par exemple est un *array* de 3 éléments `(r,g,b)` :

```python
im[200,200]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Oui on peut voir l'image aussi
<!-- #endregion -->

```python
plt.imshow(im) 
```

<!-- #region slideshow={"slide_type": "subslide"} -->
si je ne prends que la valeur de R dans RGB j'obtiens des niveaux de gris (ça marche aussi pour G ou
B)
<!-- #endregion -->

```python
plt.imshow(im[:,:,0])
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Magie
<!-- #endregion -->

```python
plt.imshow(im[:,:,0], cmap=plt.get_cmap('gray'))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Si vous voulez en savoir plus je vous invite à consulter les pages suivantes :

- <https://matplotlib.org/tutorials/introductory/images.html>
- <https://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## S'entraîner avec NumPy

Pour vous entraîner à manipuler des *arrays* et découvrir les fonctions de NumPy. Je vous recommande
la série d'exercices corrigés à <https://www.w3resource.com/python-exercises/numpy/index-array.php>.
Essayez au maximum de les résoudre sans écrire de boucles.
<!-- #endregion -->