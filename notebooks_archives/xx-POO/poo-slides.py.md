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

<!-- LTeX: language=fr -->

<!-- #region slideshow={"slide_type": "-"} -->
Cours 3 : POO
=============

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Au commencement
Au commencement étaient les variables
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
x = 27
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Elles représentaient parfois des concepts sophistiqués
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
import math

point_1 = (27, 13)
point_2 = (19, 84)

def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

distance(point_1, point_2)
```

<!-- #region slideshow={"slide_type": "-"} -->
Et c'était pénible à écrire et à comprendre
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Pour simplifier, on peut nommer les données contenues dans variables, par exemple avec un `dict`
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
point_1 = {'x': 27, 'y': 13}
point_2 = {'x': 19, 'y': 84}

def distance(p1, p2):
    return math.sqrt((p2['x']-p1['x'])**2+(p2['y']-p1['y'])**2)

distance(point_1, point_2)
```

<!-- #region slideshow={"slide_type": "-"} -->
Et c'est toujours aussi pénible à écrire mais un peu moins à lire
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut avoir une syntaxe plus agréable en utilisant des tuples nommés
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
from collections import namedtuple
Point = namedtuple('Point', ('x', 'y'))

point_1 = Point(27, 13)
point_2 = Point(19, 84)

def distance(p1, p2):
    return math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)

distance(point_1, point_2)
```

<!-- #region slideshow={"slide_type": "-"} -->
Voilà, le cours est fini, bonnes vacances.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Peut mieux faire

- Les trucs créés via `namedtuple` sont ce qu'on appelle des *enregistrements* (en C des *struct*s)
- Ils permettent de regrouper de façon lisibles des données qui vont ensemble
- Abscisse et ordonnée d'un point
- Année, mois et jour d'une date
- ~~Signifiant et signifié~~ Prénom et nom d'une personne
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
- Leur utilisation (comme tout le reste d'ailleurs) est **facultative** : on vit très bien en
  assembleur
- Mais ils permettent de rendre le code bien plus lisible (et écrivable)
- Et ils sont rétrocompatibles avec les tuples normaux
<!-- #endregion -->

```python
def mon_max(lst):
    """Renvoie le maximum d'une liste et son indice."""
    res, arg_res = lst[0], 0
    for i, e in enumerate(lst[1:], start=1):
        if e > res:
            res = e
            arg_res = i
    return res, arg_res
            
def bidule(lst1, lst2):
    return lst2[mon_max(lst1)[1]]

bidule([2,7,1,3], [1,2,4,8])
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Si on convertit `mon_max` pour renvoyer un tuple nommé, on peut continuer à utiliser `bidule`
<!-- #endregion -->

```python
MaxRet = namedtuple('MaxRet', ('value', 'indice'))
def mon_max(lst):
    """Renvoie le maximum d'une liste et son indice."""
    res, arg_res = lst[0], 0
    for i, e in enumerate(lst[1:], start=1):
        if e > res:
            res = e
            arg_res = i
    return MaxRet(res, arg_res)

def bidule(lst1, lst2):
    """Renvoie la valeur de lst2 à l'indice où lst1 atteint son max"""
    return lst2[mon_max(lst1)[1]]

bidule([2,7,1,3], [1,2,4,8])
```

```python

```

Vous êtes **fortement** encouragé⋅e⋅s à utiliser des tuples nommés quand vous écrivez une fonction qui renvoie plusieurs valeurs.

```python slideshow={"slide_type": "subslide"}
Vecteur = namedtuple('Vecteur', ('x', 'y'))

v1 = Vecteur(27, 13)
v2 = Vecteur(1, 0)

def norm(v):
    return math.sqrt(v.x**2 + v.y**2)

def is_unit(v):
    return norm(v) == 1

print(is_unit(v1))
print(is_unit(v2))
```

<!-- #region slideshow={"slide_type": "-"} -->
C'est plutôt lisible
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Mais si je veux pouvoir faire aussi de la 3d
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
Vecteur3D = namedtuple('Vecteur3D', ('x', 'y', 'z'))

u1 = Vecteur3D(27, 13, 6)
u2 = Vecteur3D(1, 0, 0)

def norm3d(v):
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def is_unit3d(v):
    return norm3d(v) == 1

print(is_unit3d(u1))
print(is_unit3d(u2))
```

C'est affreusement pénible de réécrire comme ça le même code.

<!-- #region slideshow={"slide_type": "subslide"} -->
Une autre solution
<!-- #endregion -->

```python
def norm(v):
    if isinstance(v, Vecteur3D):
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)
    elif isinstance(v, Vecteur):
        return math.sqrt(v.x**2 + v.y**2)
    else:
        raise ValueError('Type non supporté')

def is_unit(v):
    return norm(v) == 1

print(is_unit(v1))
print(is_unit(u2))
```

<!-- #region slideshow={"slide_type": "-"} -->
C'est un peu mieux mais pas top. (Même si on aurait pu trouver une solution plus intelligente)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Ces fameux objets
Une des solutions pour faire mieux c'est de passer à la vitesse supérieure : les objets.

Ça va d'abord être un peu plus désagréable, pour ensuite être beaucoup plus agréable.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
import math

class Vecteur:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

v1 = Vecteur(27, 13)
v2 = Vecteur(1, 0)

v1.x
#print(v2.norm())
```

```python slideshow={"slide_type": "subslide"}
class Vecteur3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

u1 = Vecteur3D(27, 13, 6)
u2 = Vecteur3D(1, 0, 0)

print(u1.norm())
print(u2.norm())
```

```python slideshow={"slide_type": "subslide"}
def is_unit(v):
    return v.norm() == 1

print(is_unit(v1))
print(is_unit(u2))
```

<!-- #region slideshow={"slide_type": "-"} -->
Le choix de la bonne fonction `norme` se fait automagiquement
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Résumons
  - Un objet, c'est un bidule qui regroupe
    - Des données (on dit *attributs* ou *propriétés*)
    - Des fonctions (on dit des *méthodes*)
  - Ça permet d'organiser son code de façon plus lisible et plus facilement réutilisable (croyez moi sur parole)
  
Et vous en avez déjà rencontré plein
<!-- #endregion -->

```python
print(type('abc'))
print('abc'.islower())
```

Car en Python, tout est objet. Ce qui ne veut pas dire qu'on est obligé d'y faire attention…

<!-- #region slideshow={"slide_type": "slide"} -->
## POO

La programmation orientée objet (POO) est une manière de programmer différente de la programmation procédurale vue jusqu'ici.

- Les outils de base sont les objets et les classes
- Un concept → une classe, une réalisation concrète → un objet

C'est une façon particulière de résoudre les problèmes, on parle de *paradigme*, et il y en a d'autres
  
- Fonctionnel : les outils de base sont les fonctions
- Impérative : les outils de base sont les structures de contrôle (boucles, tests…)

Python fait partie des langages multi-paradigmes : on utilise le plus pratique, ce qui n'est pas sans déplaire aux puristes mais

« *We are all consenting adults here* »
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Classes
- On définit une classe en utilisant le mot-clé `class`
- Par conventions, les noms de classe s'écrivent avec des  majuscules (CapWords convention)

<!-- #endregion -->

```python slideshow={"slide_type": "-"}
class Word:
    """ Classe Word : définit un mot de la langue """
    pass
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Pour créer un objet, on appelle simplement sa classe comme une fonction
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
word1 = Word()
print(type(word1)) # renvoie la classe qu'instancie l'objet
```

<!-- #region slideshow={"slide_type": "-"} -->
On dit que `word1` est une *instance* de la classe `Word`
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Et il a déjà des attributs et des méthodes
<!-- #endregion -->

```python
word1.__doc__
```

```python
print(dir(word1))
```

Et aussi un identifiant unique

```python
id(word1)
```

```python
word2 = Word()
id(word2)
```

<!-- #region slideshow={"slide_type": "subslide"} -->

## Constructeur et attributs

- Il existe une méthode spéciale `__init__()` qui automatiquement appelée lors de la création d'un
  objet. C'est le constructeur
- Le constructeur permet de définir un état initial à l'objet, lui donner des attributs par exemple
- Les attributs dans l'exemple ci-dessous sont des variables propres à un objet, une instance

<!-- #endregion -->

```python slideshow={"slide_type": "-"}
class Word:
    """ Classe Word : définit un mot de la langue """
    
    def __init__(self, form, lemma, pos):
        self.form = form
        self.lemma = lemma
        self.pos = pos

word = Word('été', 'être', 'V')
word.lemma
```

```python
word2 = Word('été', 'été', 'NOM')
word2.lemma
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Méthodes

- Les méthodes d'une classe sont des fonctions. Elles indiquent quelles actions peut mener un objet,
  elles peuvent donner des informations sur l'objet ou encore le modifier.
- Par convention, on nomme `self` leur premier paramètre, qui fera référence à l'objet lui-même.

<!-- #endregion -->

```python slideshow={"slide_type": "-"}
class Word:
    """ Classe Word : définit un mot simple de la langue """
    def __init__(self, form, lemma, pos):
        self.form = form
        self.lemma = lemma
        self.pos = pos
    
    def __repr__(self):
        return f"{self.form}"
    
    def brown_string(self):
        return f"{self.form}/{self.lemma}/{self.pos}"
    
    def is_inflected(self):
        """
        Returns True is the word is inflected
        False otherwise
        """
        if self.form != self.lemma:
            return True
        else:
            return False

w = Word('orientales', 'oriental', 'adj')
print(w)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Pourquoi `self` ? Parce que écrire `w.is_inflected()` c'est du sucre pour
<!-- #endregion -->

```python
Word.is_inflected(w)
```

<!-- #region slideshow={"slide_type": "slide"} -->
# Héritage

<!-- #endregion -->

```python slideshow={"slide_type": "-"}
class Cake:
    """ un beau gâteau """

    def __init__(self, farine, oeuf, beurre):
        self.farine = farine
        self.oeuf = oeuf
        self.beurre = beurre
        self.poids = self.farine + self.oeuf*50 + self.beurre

    def is_trop_gras(self):
        if self.farine + self.beurre > 500:
            return True
        else:
            return False
    
    def cuire(self):
        return self.beurre / self.oeuf
```

```python
gateau = Cake(200, 3, 800)
gateau.poids
```

<!-- #region slideshow={"slide_type": "-"} -->
Cake est la classe mère.

Les classes enfants vont hériter de ses méthodes et de ses attributs.

Cela permet de factoriser le code, d'éviter les répétitions et les erreurs qui en découlent.

<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
class CarrotCake(Cake):
    """ pas seulement pour les lapins
        hérite de Cake """

    carotte = 3
    
    def cuire(self):
        return self.carotte * self.oeuf
    
class ChocolateCake(Cake):
    """ LE gâteau 
        hérite de Cake """
        
    def is_trop_gras(self):
        return False
```

```python slideshow={"slide_type": "-"}
gato_carotte = CarrotCake(200, 3, 150)
gato_carotte.cuire()
```

```python
gato_1.cuire()
```

```python
gato_2 = ChocolateCake(200, 6, 600)
gato_2.is_trop_gras()
```

L'héritage est à utiliser avec parcimonie. On utilisera volontiers par contre la composition c-a-d
l'utilisation d'objets d'autres classes comme attributs. Voir
<https://python-patterns.guide/gang-of-four/composition-over-inheritance/>


### ☕  Exos 1 ☕

Écrire une classe `Sentence` et une classe `Word` qui représenteront les données d'un fichier ud (https://universaldependencies.org/).  
Vous écrirez également un programme qui reçoit un fichier .conll en argument et instancie autant d'objets Sentence et Word que nécessaires.


### ☕  Exos 2 ☕

Écrivez un script qui reçoit trois arguments : un répertoire de fichiers conllu, une chaîne de car. notant le mode (form ou pos) et un entier (n entre 2 et 4).  
Votre calculera les fréquences des n-grammes (où la valeur de n est passée en argument) dans les fichiers du répertoire. Deux modes de calcul : par forme ou par pos.  
Je veux juste votre script, pas les données, ni les résultats.

