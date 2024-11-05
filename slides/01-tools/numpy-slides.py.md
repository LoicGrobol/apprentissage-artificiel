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
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
<!-- LTeX: language=fr -->


TP 2 : NumPy
=================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## NumPy ?

[NumPy](https://numpy.org/).

NumPy est un des packages les plus utilisés de Python. Il ajoute au langage des maths plus
performantes, le support des tableaux multidimensionnels (`ndarray`) et du calcul matriciel.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Installation

Si vous n'avez pas déjà installé le `requirements.txt` du cours, installez Numpy **dans votre
environnement virtuel** avec `python -m pip install numpy`.

<!-- #region slideshow={"slide_type": "subslide"} -->
On importe Numpy comme ceci
<!-- #endregion -->

```python
import numpy as np
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Ne faites pas autrement, ne lui donnez pas d'autre nom que le conventionnel `np`, sinon ça rendra
votre code plus désagréable pour tout le monde.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Maths de base
<!-- #endregion -->

A priori rien de bien extraordinaire.

Numpy vous donne accès à des fonctions mathématiques, souvent plus efficaces que les
équivalents du module standard `math`, plus variées et apportant souvent d'autres avantages, comme
une meilleure stabilité numérique.

```python
np.log(1.5)
```

```python
np.sqrt(2)
```

```python
np.logaddexp(2.7, 1.3)
```

Comment on les apprend ? **En allant lire la [doc](https://numpy.org/doc/stable/). Par exemple pour
[`np.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html#numpy.sqrt).

Ça parle d'`array`, on explique ça dans une grosse minute.

On peut aussi construire directement des nombres

```python
x = np.float64(27.13)  # on revient dans un instant sur pourquoi 64
y = np.float64(3.14)
```

qui s'utilisent exactement comme les nombres de Python.

```python
x+y
```

```python
x*y
```

etc.

<!-- #region slideshow={"slide_type": "slide"} -->
### Précision
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
# Un nombre à virgule flottante codé sur 64 bits, comme ceux par défaut
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
Et même les combiner avec les types habituels de Python (qui sont considérés comme des `float64`)
<!-- #endregion -->

```python
print(double + 1.0, type(double + 1.0))
```

<!-- #region slideshow={"slide_type": "slide"} -->
## `ndarray`

Le principal intérêt de Numpy ce sont les `array` (classe `ndarray`), à une dimension (vecteurs), deux
dimensions (matrices) ou trois et plus (« tenseurs »).

Un `array` sera en général plus rapide et plus compact (moins de taille en mémoire) qu'une liste Python.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
NumPy ajoute plein de fonctions pour manipuler ses `array` de façon optimisée. Il est recommandé
d'utiliser ces fonctions plutôt que des boucles, qui seront en général beaucoup plus lentes.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut créer un `array` à partir d'une liste (ou d'un tuple) :
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
**Mais** à la différence d'une liste, un `array` aura les caractéristiques suivantes :

- Une taille fixe (donnée à la création).
- Ses éléments doivent tous être de même type.

(C'est ça qui permet d'optimiser les opérations et la mémoire)
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} tags=["raises-exception"]
b.append(1)
```

```python slideshow={"slide_type": "subslide"}
a = np.array([1, 1.2])
a
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Opérations
<!-- #endregion -->

On peut se servir des `array`s pour faire des maths.

### Opérations classiques

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

### ufunc

C'est un des points les plus important de Numpy : ses fonctions opèrent sur les `array`s coordonnée
par coordonnée :

```python
a = np.array([5, 6, 7, 8, 9, 10])
a
```

```python
np.sqrt(a)
```

Pourquoi c'est si intéressant ? Parce que comme ces opérations traitent les coordonnées de façon
indépendantes, elles peuvent se faire **en parallèle** : dans Numpy il n'y a pas de boucle Python
qui traite les coordonnées une par une, mais des implémentations qui exploitent entre autres les
capacités de votre machine pour faire tous ces calculs simultanément, ce qui va beaucoup, beaucoup
plus vite.

Ces fonctions, définies pour opérer sur une coordonnée, et qui peuvent se paralléliser à l'échelle
d'un `array` s'appellent des [`ufunc`](https://numpy.org/doc/stable/reference/ufuncs.html)
(*Universal FUNctions*), Numpy en fournit beaucoup par défaut, et c'est en général mieux de les
utiliser, mais vous pouvez aussi définir les vôtres si vous avez vraiment un besoin particulier.

TL;DR: **ne traitez les `array`s avec des boucles que si vous n'avez vraiment pas le choix.**

<!-- #region slideshow={"slide_type": "subslide"} -->
### Opérations sur plusieurs `array`s
<!-- #endregion -->

Les opérateurs usuels, comme le reste, agissent en général coordonnée par coordonnée :

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
a / c
```

```python
a / 1000
```

Tiens, celui-ci est un peu curieux : vous voyez pourquoi ?

<!-- #region slideshow={"slide_type": "subslide"} -->
Produit matriciel : c'est l'exception !
<!-- #endregion -->

```python
m1 = np.array([[1, 2],[ 3, 4], [5, 6]])
m1
```

```python
m2 = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
m2
```

·

```python
np.matmul(m1, m2)
```

On peut aussi utiliser l'opérateur `@`

```python
m1@m2
```

<!-- #region slideshow={"slide_type": "fragment"} -->
- Transposition (échanger lignes et colonnes)
<!-- #endregion -->

```python
m1
```

```python
m1.T
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Créer un `array`

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

```python
np.ones(3, dtype=np.float32)
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
- `np.linspace(start, stop)` (crée un `array` avec des valeurs réparties uniformément entre `start`
   et `stop` (50 par défaut))
<!-- #endregion -->

```python
np.linspace(0, 10, num=5)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
- `np.empty` (crée un `array` vide, ou plus précisément avec des valeurs non-initialisées)
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
Allez lire [la doc](https://numpy.org/doc) !
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Infos sur les `ndarray`

Pour avoir des infos sur les `array` que vous manipulez vous avez :

- `dtype` (type des éléments)
<!-- #endregion -->

```python
b.dtype
```

- `shape` (un tuple avec la taille de chaque dimension)

```python
b.shape
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `size` (le nombre total d'éléments)
<!-- #endregion -->

```python
b.size
```

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
a[2:5]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- Au-delà d'une dimension il y a une syntaxe différente
<!-- #endregion -->

```python
b = np.random.randint(13, 27, size=(5, 7))
b
```

```python slideshow={"slide_type": "fragment"}
b[1, 2]
```

```python
b[1, 2:5]
```

```python slideshow={"slide_type": "fragment"}
b[1, :] # 2e ligne, toutes les colonnes
```

```python
b[1, ...]
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
- On peut aussi faire des sélections avec des conditions (comme dans pandas et co.)
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
## Broadcasting

Bon à savoir :
<!-- #endregion -->

```python
a = np.array([[1, 2, 3], [5, 6, 7]])
a
```

```python
c = np.array([2, 4, 8])
c
```

```python
a + c
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Ce qui se passe : si un des tableaux a moins de dimensions que l'autre, Numpy fait automatiquement
la conversion pour que tout se passe comme si on avait ajouté des dimensions par copie :
<!-- #endregion -->

```python
np.broadcast_to(c, [2, 3])
```

C'est aussi ce qui se passait tout à l'heure avec

```python
a + 1
```

Ici `1` est considéré comme un tableau de `shape` `[0]`, et broadcasté en conséquence.

Attention, ça ne marche que si les dimensions sont compatibles : ici il faut que `c` et `a` aient le
même nombre de colonnes.

<!-- #region slideshow={"slide_type": "subslide"} -->
Pensez à [lire la doc](https://numpy.org/doc/stable/user/basics.broadcasting.html).

<!-- #region slideshow={"slide_type": "slide"} -->
## Changer de forme
<!-- #endregion -->

```python
c = np.arange(6)
c
```

<!-- #region slideshow={"slide_type": "fragment"} -->
J'en fais une matrice de 2 lignes et 3 colonnes
<!-- #endregion -->

```python
d = c.reshape(2, 3)
d
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Inversement on peut tout aplatir
<!-- #endregion -->

```python
d.flatten()
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Ou ajouter des dimensions (par exemple pour guider le broadcasting, voir la doc, etc.)
<!-- #endregion -->

```python
e = c[:, np.newaxis]
e
```


```python
print(c.shape, e.shape)
```

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
En pratique, quand on veut faire des trucs un peu plus complexes, il vaut mieux utiliser des surcouches comme [`plotnine`](https://plotnine.org/). On en reparlera si on a le temps.

Juste un exemple avec une image pour la culture :
<!-- #endregion -->

`plt.imread` permet de changer un fichier image en objet python… devinez lequel

```python
im = plt.imread("nimona_u_00_24_12_08-1280.jpg")
type(im)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Bingo, un `array`. En même temps, c'est jamais qu'une matrice de pixels, une image.
<!-- #endregion -->

```python
im.shape
```

Un `array` à trois dimensions : X, Y (les coordonnées du pixel) et la valeur RGB du pixel (trois valeurs).

Le pixel `(200, 200)` par exemple est un `array` de 3 éléments `(r, g, b)` :

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
si je ne prends que la valeur de R dans RGB :
<!-- #endregion -->

```python
plt.imshow(im[:, :, 0])
```

<!-- #region slideshow={"slide_type": "subslide"} -->
qu'est-ce qui se passe ici ?
<!-- #endregion -->

```python
plt.imshow(im[:, :, 0], cmap=plt.get_cmap('gray'))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Si vous voulez en savoir plus je vous invite à consulter les pages suivantes :

- <https://matplotlib.org/tutorials/introductory/images.html>
- <https://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/>
<!-- #endregion -->


<!-- #region slideshow={"slide_type": "slide"} -->
## Plus de lecture

Il y a beaucoup à dire sur Numpy, aussi bien dans l'étendue de ses fonctionnalités que pour les détails de son implémentations.

Si vous voulez tirer au mieux partie de votre machine et rendre votre code plus efficace (et vous **devez** le vouloir), plus vous en saurez sur NumPy, mieux ce sera.

Comme introduction à l'implémentations de NumPy et à pourquoi c'est tellement plus efficace que du Python pur, je vous renvoie à l'excellent [Performance tips of NumPy arrays](https://shihchinw.github.io/2019/03/performance-tips-of-numpy-ndarray.html). Prenez le temps de le lire attentivement, de le comprendre, de le mémoriser, de me poser des questions dessus et d'aller lire tous les articles cités dans la bibliographie (oui, tous, et en particulier [Advanced Numpy](https://scipy-lectures.org/advanced/advanced_numpy/).

Revenez-y fréquement et ne cessez jamais de vous demander comment marchent les bibliothèques que vous utilisez.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## S'entraîner avec NumPy

À vous de jouer maintenant. Allez à <https://www.w3resource.com/python-exercises/numpy/index.php> et
faites autant d'exercices que possible.

Essayez au maximum de les résoudre sans écrire de boucles. Utilisez la doc au maximum, si vous ne
réussissez pas un exercice, assurez-vous de complètement comprendre la solution avant de passer à la
suite.
<!-- #endregion -->
