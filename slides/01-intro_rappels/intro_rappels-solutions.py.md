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

Cours 1 : corrections
=====================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22

## ✍️ Exos 1 ✍️

### Carré

Rien de mystérieux ici

```python
def square(num):
    """Renvoie le nombre donné en argument au carré."""
    return num**2
```

```python
assert square(3) == 9
assert square(0) == 0
assert square(-2) == 4
```

### Parité

Version élémentaire

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    if num % 2 == 0:
        return True
    else:
        return False
```

On peut utiliser `return` comme un court-circuit

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    if num % 2 == 0:
        return True
    return False
```

Ou simplement utiliser le fait que la comparaison est déjà un booléen

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    return num % 2 == 0
```

En poussant le golf plus loin : en exploitant le fait que `0` est faux.

Ou simplement utiliser le fait que la comparaison est déjà un booléen

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    return not (num % 2)
```

```python
assert is_even(1) == False
assert is_even(2) == True
assert is_even(-3) == False
assert is_even(-42) == True
assert is_even(0) == True
```

## ✍️ Exo 2 ✍️

On peut faire comme ça, mais c'est trop verbeux

```python
def on_fait_la_taille(moi, toi):
    """Dis si moi est plus grand que toi"""
    if moi > toi:
      return "plus grand"
    else:
      if moi < toi:
        return "plus petit"
      else:
        return "pareil"
```

C'est mieux comme ça

```python
def on_fait_la_taille(moi, toi):
    """Dis si moi est plus grand que toi"""
    if moi > toi:
      return "plus grand"
    elif moi < toi:
      return "plus petit"
    else:
      return "pareil"
```

On peut aussi utiliser des courts-circuits (voir [exo 1](#✍️-Exos-1-✍️)) mais je trouve ça moins
clair.

```python
assert on_fait_la_taille(100, 80) == "plus grand"
assert on_fait_la_taille(100, 120) == "plus petit"
assert on_fait_la_taille(100, 100) == "pareil"
```

## ✍️ Exo 3 ✍️

Vous reprenez votre fonction `is_even` de façon à afficher "Erreur de type" quand l'argument n'est
pas de type `int`

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    if not isinstance(num, int):
      return "Erreur de type"
    return num % 2 == 0
```

```python
assert is_even(1) == False
assert is_even(2) == True
assert is_even(-3) == False
assert is_even(-42) == True
assert is_even(0) == True
assert is_even("test") == "Erreur de type"
```

## ✍️ Exo 4 ✍️

```python
def say_hello(firstname, lastname):
    return f"Hello {firstname} {lastname} !"
```

```python
assert say_hello("Lucky", "Luke") == "Hello Lucky Luke !"
```

## ✍️ Exo 5 ✍️

```python
def change_char(s, idx):
    """In the given string, change the char at given index for 'z' and return the modified str
    ex: change("maison", 2) -> mazson
    """
    exploded = list(s)
    exploded[idx] = "z"
    return "".join(exploded)
```

```python
assert isinstance(change_char("maison", 3), str)
assert change_char("maison", 3) == "maizon"
assert change_char("maison", 0) == "zaison"
```

## ☕ Exos 6 ☕

Il y a plus élémentaire, mais ça se fait bien avec des compréhensions

```python
def fr_ar(s):
    """
    recherche les pronoms personnels dans la chaîne donnée en argument
    renvoie leurs équivalents en arabe sous forme de liste
    """
    # from https://fr.wikipedia.org/wiki/Liste_Swadesh_de_l%27arabe and https://fr.wiktionary.org/wiki/هُمَا
    fr_ar_dict = {'je':'أنا', 'tu':'أنت', 'il': 'هو', 'elle': 'هي', 'iel': 'هما', 'nous': 'نحن', 'vous': 'انتما', 'ils': 'هما', 'elles': 'هنَّ', 'iels': 'هما'}
    res = []
    for w in s.split():
        trad = fr_ar_dict.get(w)
        if trad is not None:
            res.append(trad)
    return res
```

Voire un peu plus
[ésotérique](https://docs.python.org/3/reference/expressions.html?highlight=walrus#assignment-expressions) :

```python
def fr_ar(s):
    """
    recherche les pronoms personnels dans la chaîne donnée en argument
    renvoie leurs équivalents en arabe sous forme de liste
    """
    # from https://fr.wikipedia.org/wiki/Liste_Swadesh_de_l%27arabe and https://fr.wiktionary.org/wiki/هُمَا
    fr_ar_dict = {'je':'أنا', 'tu':'أنت', 'il': 'هو', 'elle': 'هي', 'iel': 'هما', 'nous': 'نحن', 'vous': 'انتما', 'ils': 'هما', 'elles': 'هنَّ', 'iels': 'هما'}
    res = []
    for w in s.split():
        if (trad := fr_ar_dict.get(w)) is not None:
            res.append(trad)
    return res
```

Un bon tutoriel sur l'opérateur morse `:=` sur
[RealPython](https://realpython.com/python-walrus-operator/) (en anglais).

```python
assert fr_ar("trop bizarre cet exercice") == []
assert fr_ar("iel nous a rien dit") == ['هما', 'نحن']
```

Dans tous les cas, il fallait se méfier de l'exemple : dans beaucoup d'éditeurs, à cause du
changement de direction, la liste apparait dans le désordre !

### 1. Des triangles

1. Écrire une fonction `la_plus_grande(longueur1, longueur2, longueur3)` qui renvoie la longueur du
   plus grand côté (une fonction de python fait peut-être déjà cela...).
2. Écrire une fonction `est_equilateral(longueur1, longueur2, longueur3)` qui détermine si un
   triangle est équilatéral ou non (les trois côtés ont la même longueur).
3. Écrire une fonction `est_isocele(longueur1, longueur2, longueur3)` qui détermine si un triangle
   est isocèle (deux côtés de même longueur, mais pas trois) ou non.
4. Écrire une fonction `caracteristiques(longueur1, longueur2, longueur3)` qui renvoie la nature et
   la taille du plus grand côté d'un triangle. On dira qu'un triangle est `quelconque` s'il n'est ni
   équilatéral ni isocèle. Affiche `pas un triangle` si les longueurs données ne font pas un
   triangle (la longueur du plus grand côté est supérieure à celle des deux autres).

```python
def la_plus_grande(longueur1, longueur2, longueur3):
    """Renvoie la plus grande longueur."""
    return max(longueur1, longueur2, longueur3)

def est_equilateral(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est équilatéral."""
    return longueur1 == longueur2 and longueur2 == longueur3

def est_isocele(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est isocele."""
    deux_egales = longueur1 == longueur2 or longueur1 == longueur3 or longueur2 == longueur3
    return deux_egales and not est_equilateral(longueur1, longueur2, longueur3)

def est_triangle(longueur1, longueur2, longueur3):
    """Renvoie si les longueurs données font bien un triangle."""
    maxi = la_plus_grande(longueur1, longueur2, longueur3)
    somme = longueur1 + longueur2 + longueur3
    return maxi <= (somme - maxi)  # la somme des deux côtés est au plus maxi

def caracteristiques(longueur1, longueur2, longueur3):
    """Affiche les caractéristiques d'un triangle.
    Les caractéristiques d'un triangle sont :
        - sa nature
        - la taille de son plus grand côté.

    On dira qu'un triangle est `quelconque` s'il n'est ni équilatéral ni isocèle.

    Affiche `pas un triangle` si les longueurs données ne font pas un triangle
    (la longueur du plus grand côté est supérieure à celle des deux autres).
    """
    if not est_triangle(longueur1, longueur2, longueur3):
        return "pas un triangle"
    else:
        maxi = la_plus_grande(longueur1, longueur2, longueur3)
        if est_equilateral(longueur1, longueur2, longueur3):
            return ("equilatéral", maxi)
        elif est_isocele(longueur1, longueur2, longueur3):
            return ("isocèle", maxi)
        else:
            return "quelconque", maxi
```

```python
assert caracteristiques(1, 1, 1) ==  ("equilatéral", 1)
assert caracteristiques(1, 1, 2) == ("isocèle", 2)
assert caracteristiques(1, 2, 1) == ("isocèle", 2)
assert caracteristiques(2, 1, 1) == ("isocèle", 2)
assert caracteristiques(2, 3, 1) == ("quelconque", 3)
assert caracteristiques(2, 3, 6) == "pas un triangle"
assert caracteristiques(6, 3, 2) == "pas un triangle"
assert caracteristiques(2, 6, 3) == "pas un triangle"
```

### 2. Des heures

1. Écrire une fonction `heures(secondes)` qui prend un nombre de secondes (entier) et le convertit
   en heures, minutes et secondes sous le format `H:M:S` où `H` est le nombre d'heures, `M` le
   nombre de minutes et `S` le nombre de secondes.
2. Écrire une fonction `secondes(heure)` qui prend une heure au format `H:M:S` et renvoie le nombre
   de secondes correspondantes (entier).

On ne gèrera ici pas les cas incohérents comme un nombre de secondes négatif ou une heure mal formatée.

```python
def heures(secondes):
    """Prend un nombre de secondes (entier) et le convertit en heures, minutes
    et secondes sous le format `H:M:S` où `H` est le nombre d'heures,
    `M` le nombre de minutes et `S` le nombre de secondes.

    On suppose que secondes est positif ou nul (secondes >= 0).
    """
    H = secondes // 3600
    M = (secondes % 3600) // 60
    S = secondes % 60
    return f"{H}:{M}:{S}"

def secondes(heure):
    """Prend une heure au format `H:M:S` et renvoie le nombre de secondes
    correspondantes (entier).

    On suppose que l'heure est bien formattée. On aura toujours un nombre
    d'heures valide, un nombre de minutes valide et un nombre de secondes valide.
    """
    H, M, S = heure.split(":")
    return (3600 * int(H)) + (60 * int(M)) + int(S)
```

```python
assert (heures(0)) == "0:0:0"
assert(heures(30)) == "0:0:30"
assert(heures(60)) == "0:1:0"
assert(heures(66)) == "0:1:6"
assert(heures(3600)) == "1:0:0"
assert(heures(86466)) == "24:1:6"
assert(secondes('0:0:0')) == "0"
assert(secondes('6:6:6')) == "21966"
assert(secondes(heures(86466))) == "86466"
assert(heures(secondes('24:1:1'))) == "24:1:1"
```

### 3. Des cartes

Nous jouons aux cartes à quatre personnes. On appelle un pli l'ensemble des cartes jouées dans un
tour (ici, quatre cartes). Chaque carte a une valeur (un entier de 1 à 13). Chaque carte a également
une couleur : carreau, trèfle, cœur ou pic. Ces couleurs sont notés avec une lettre: carreau=`D`,
trèfle=`C`, cœur=`H` et pic=`S`. Une carte est alors une chaîne avec sa couleur et sa valeur, par
exemple l'as de pic est noté `S1`, la dame de cœur `H12`. La carte du premier joueur `carte1` donne
la couleur attendue. Une carte qui n'est pas à la bonne couleur perd automatiquement. Écrire une
fonction `gagne_couleur(carte1, carte2, carte3, carte4)` qui renvoie la carte qui remporte le pli en
faisant attention aux couleurs.  

On ne gèrera pas certains cas incohérents comme une carte ou un pli invalide.

```python
def gagne_couleur(carte1, carte2, carte3, carte4):
    """Affiche la carte qui remporte le pli en faisant attention aux couleurs :
        - la carte du premier joueur `carte1` donne la couleur attendue.
        - une carte qui n'est pas à la bonne couleur perd automatiquement.

    On ne gèrera pas certains cas incohérents comme une carte ou un pli invalide.
    """
    def gagne(carte1, carte2, couleur):  # on peut aussi définir une fonction dans une fonction
        if carte2[0] != couleur:
            return carte1
        elif int(carte1[1:]) < int(carte2[1:]):  # carte1 est forcément valide, pas besoin de vérifier
            return carte2
        else:
            return carte1
    couleur = carte1[0]
    gagnante = carte1
    for carte in [carte2, carte3, carte4]:
        gagnante = gagne(gagnante, carte, couleur)
    return gagnante
```

```python
assert(gagne_couleur('S1', 'S2', 'S3', 'S4')) == 'S4'
assert(gagne_couleur('S4', 'S3', 'S2', 'S1')) == 'S4'
assert(gagne_couleur('S1', 'D2', 'C3', 'H4')) == 'S1'
assert(gagne_couleur('S1', 'D2', 'S13', 'S10')) == 'S13'
```
