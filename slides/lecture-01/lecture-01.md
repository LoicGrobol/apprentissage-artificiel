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

Cours 1 : Introduction et rappels Python
========================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22


## Bonjour

- Loïc Grobol (il/iel) [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)
- PHILLIA / MoDyCo (Bâtiment Rémond, 4ème, bureau 404C)
- *Office hours* le mardi après-midi, n'hésitez pas à passer y compris sans rendez-vous (mais je
  préfère si vous m'envoyez un mail pour me prévenir)
- De manière générale, n'hésitez pas à m'écrire


## Infos pratiques

- **Quoi** « Apprentissage artificiel »
- **Où** Salle 219, bâtiment Paul Ricœur
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 22/09 au 17/11
  - Voir [le
    planning](http://www.tal.univ-paris3.fr/plurital/admin/Calendrier_M2_TAL_PX_2021_22.xlsx) pour
    les dates exactes

→ PC portable obligatoire pour les cours, de préférence chargé. Si c'est un problème parlez m'en
tout de suite et on trouvera une solution.

## Liens

- La page du cours (slides, documents, nouvelles, consignes…)
  - → <https://loicgrobol.github.io/apprentissage-artificiel>
- Le dépôt GitHub (sources, compléments et historique)
  - → <https://github.com/LoicGrobol/apprentissage-artificiel>

## Setup

- Tous les notebooks s'ouvrent dans [Binder](https://mybinder.org), y compris les slides
- Pour cette séance on peut s'en contenter, pour la suite ça ne suffira pas
- Pour la séance prochaine, il faudra avoir Python 3 installé (mais c'est déjà votre cas à toustes,
  non ?)


## Programme du cours

- Programmer en Python
  - On va mettre l'accent sur le concret
  - Bonnes pratiques, outils de développement
- Interfaces web, plutôt côté programme
  - Réseau
  - API : en utiliser et en faire
  - Un peu de sites web

## Aujourd'hui

Crash course Python

**C'est parti**

```python
from IPython.display import display
```

## Les opérateurs mathématiques

`+` addition
`-` soustraction  
`*` multiplication  
`/` division  
`//` la division entière  
`%` modulo (reste de la division)  
`**` puissance  

- L'ordre des opérations est l'ordre classique en mathématiques (puissance passe avant les
  opérations).
- On peut utiliser des parenthèses pour définir des priorités.


### ✍️ Exos 1 ✍️

C'est à vous de jouer !

Vous avez une fonction à compléter (ça vous apprendra à écrire des fonctions 😤).  

À chaque fois j'essaierai d'ajouter une cellule avec des tests qui vous permettront de valider votre
code. Écrivez votre code dans la cellule de la fonction (et enlevez `pass`), exécutez cette cellule
(bouton 'Run' ou ctrl + Enter) puis exécutez la cellule de test.

L'objectif est que vous soyez autonome pour valider ces exos (et accessoirement de vous familiariser
avec les tests).

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    pass # Votre code ici
```

```python
assert is_even(1) == False
assert is_even(2) == True
assert is_even(-3) == False
assert is_even(-42) == True
assert is_even(0) == True
```

```python
def square(num):
    """Renvoie le nombre donné en argument au carré."""
    pass # Votre code ici
```

```python
assert square(3) == 9
assert square(0) == 0
assert square(-2) == 4
```

## Opérateurs de comparaison

- `<` inférieur  / `<=` inférieur ou égal
- `>` supérieur  / `>=` supérieur ou égal
- `==` égal / `!=` différent
- `is` identité (pour les objets surtout)/ `is not` non identité

### ✍️ Exo 2 ✍️

```python
def on_fait_la_taille(moi, toi):
    """Vrai ssi `moi` est plus grand que `toi`""""
    pass # Votre code ici
```

```python
assert on_fait_la_taille(100, 80) == "plus grand"
assert on_fait_la_taille(100, 120) == "plus petit"
assert on_fait_la_taille(100, 100) == "pareil"
```

### Identité et égalité

`a == b` est vrai si `a` et `b` sont égaux, `a is b` si c'est le même objet.

```python
display(["spam"] == ["spam"])
display(["spam"] is ["spam"])
```

Il y a quelques pièges, mais on y reviendra

## Les variables

- L'affectation des variables se fait à l'aide du symbole `=`  
- Si la variable est placée à droite du symbole `=`, sa *valeur* est affectée à la variable placée à
  gauche.
- Les noms de variable sont composés de caractères alphabétiques (min ou maj), des chiffres et de
  l'underscore.
- Les noms de variable sont choisis par le programmeur, ils doivent être le plus clair possible. Il
  est conseillé de suivre la [PEP 8](https://www.python.org/dev/peps/pep-0008/).

```python
spam = 3 + 2
print(spam)

eggs = spam
print(eggs)
```

```python
je-ne-suis-pas-une-variable = 2 
```

```python
3_moi_non_plus = 2 + 3
```

- Seuls les mots réservés sont interdits.

```python
import keyword
print(keyword.kwlist)
```

On *peut* faire des trucs exotiques (voir la [doc](https://docs.python.org/3/reference/lexical_analysis.html#identifiers))

```python
z̷̫̻̘̞̫͓̩̮͐̄̀̇̐̅̈́̂̊͂̚͜͝͝ā̷̧̛̻͎͙̣̻̫̹̙̠̖̬̏̈́͒͋̃́̄̿͋͛̊l̶̢̡̹̠̜͉̗̆̎̐̀͑͗̒̒́́̅̿͒͜g̶̢̡̼̭̭̫̽̄̓̌͗͠͝ó̴͇̯͚̮̟̻͕̭͂͑̅͐̿͂͗͌͌̌̓ͅ = "HE COMES"
print(z̷̫̻̘̞̫͓̩̮͐̄̀̇̐̅̈́̂̊͂̚͜͝͝ā̷̧̛̻͎͙̣̻̫̹̙̠̖̬̏̈́͒͋̃́̄̿͋͛̊l̶̢̡̹̠̜͉̗̆̎̐̀͑͗̒̒́́̅̿͒͜g̶̢̡̼̭̭̫̽̄̓̌͗͠͝ó̴͇̯͚̮̟̻͕̭͂͑̅͐̿͂͗͌͌̌̓ͅ)
```

MAIS ON NE LE FAIT PAS

# Les types

- Python est un langage à typage *dynamique* fort : le type d'une variable est déterminé par
  l'interpréteur.
- Python est un langage à typage dynamique *fort* : pas de conversion implicite, certaines actions
  sont interdites.

```python
"Hello" + 1
```

- La fonction `type()` retourne le type de la variable donnée en argument.

```python
type("Hello")
```

- La fonction `isinstance(obj, class)` vous dit si l'objet donné en argument est de la classe
  'class' ou non

```python
isinstance('hello', int)
```

### ✍️ Exo 3 ✍️

Vous reprenez votre fonction `is_even` de façon à afficher "Erreur de type" quand l'argument n'est pas de type `int`

```python
def is_even(num):
    """
    returns True is num is even, False if odd
    """
    # votre code ici
```

```python
assert is_even(1) == False
assert is_even(2) == True
assert is_even(-3) == False
assert is_even(-42) == True
assert is_even(0) == True
assert is_even("test") == "Erreur de type"
```

## Les chaînes de caractère


- Les chaînes de caractères sont entourées de quotes simples `'` ou doubles `"`
- Si votre mot contient une apostrophe, entourez-le de guillemets `"`

```python
'Ça donne une erreur t'as vu'
```

```python
"Ça donne une erreur t'as vu"
```

On peut aussi utiliser trois quotes pour avoir une chaîne de caractères sur plusieurs lignes

```python
"""Ceci est une
chaîne de caractères
sur plusieurs lignes
Je peux y mettre des simples ' et double " quotes sans problème !
"""
```

Les chaînes sont des *séquences*, on peut leur appliquer les opérations suivantes propres à la catégorie d'objets *séquences* :

(Vous connaissez d'autres *séquences* au fait ?)

- longueur, minimum, maximum

```python
spam = "bonjour"
print(len(spam))
print(max(spam))
```

- _indexing_
  - Les indices commencent à `0` !

```python
spam = "bonjour"
print(spam[2])
print(spam[-1])
```

- _slicing_
  - `spam[i:j]`, c'est `spam[i]`, `spam[i+1]`, …, `spam[j-1]`

```python
print(spam[0:3]) # 3 premiers éléments
print(spam[-3:]) # 3 derniers éléments
```

- _membership_

```python
if 'u' in spam:
    print("Il y a un u dans {}".format(var))
```

Les chaînes ont aussi des fonctions qui leur sont propres

Voir la liste complète dans la doc python

- `lower()` transforme la chaine en minuscules
- `upper()` transforme la chaine en majuscules
- `replace(old, new)` remplace les occurrences de `old` par `new`
- `strip(chars=None)` appelé sans arguments supprime le ou les espaces en tête et en fin de chaîne  
- `rstrip(chars=None)` fait la même chose en fin de chaîne uniquement
- `lstrip(chars=None)` idem en début de chaîne
- `split(sep=None)` découpe une chaîne en fonction de `sep` et renvoie une liste. Si `sep` n'est pas
  donné, coupe sur tous les caractères d'espace
- `join(iterable)` est l'inverse de `split`, il permet de joindre les éléments d'un *iterable* pour
  former une seule chaîne de caractères
  [`format()`](https://docs.python.org/3/library/string.html#formatstrings) depuis python3 (et
  python2.7) pour effectuer l'[interpolation de
  chaîne](https://en.wikipedia.org/wiki/String_interpolation)

```python
words = "bonjour ça va ?".split(' ')
"-".join(words)
```

### Formatage de chaînes

> There should be one — and preferably only one — obvious way to do it.  ([PEP 20 : *Zen of
> Python*](https://www.python.org/dev/peps/pep-0020/))

Sauf que :

- Concaténation avec `+` **à éviter**
- Interpolation avec `format()`
- [f-string](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) depuis python3.6

```python
name = "Clément"
coffee_price = 0.6

print("Tiens salut " + name + ". T'aurais pas " + str(coffee_price*2) + " euros pour 2 cafés ?")

print("Tiens salut {}. T'aurais pas {} euros pour 2 cafés ?".format(name, coffee_price*2))

print(f"Tiens salut {name}. T'aurais pas {coffee_price*2} euros pour 2 cafés ?")
```

On évite de faire ça avec `+` parce que c'est moins lisible et que c'est **lent**. De fait on créé
une chaîne intermédiaire à chaque étape.

```python
s1 = "Tiens salut " + name
s2 = s1 + ". T'aurais pas "
s3 = s2 + + str(coffee_price*2)
s4 = s3 + + " euros pour 2 cafés ?"
```

Sur l'exemple ci-dessus ça va, mais on se retrouve vite à additionner des centaines de chaînes et
c'est la galère.


Pour concaténer beaucoup de chaînes il vaut mieux utiliser `join`

```python
print(" 👏 ".join(["On", "ne", "concatène", "pas", "des", "chaînes", "de", "caractères", "avec", "+"]))
```

Attention aussi à la concaténation implicite

```python
spam = ("Hello, " "there")
ham = ("General ", "Kenobi")
print(spam)
print(ham)
```

### ✍️ Exo 4 ✍️

```python
def say_hello(firstname, lastname):
    # avec des f-string svp
    # votre code ici
    pass
```

```python
assert say_hello("Lucky", "Luke") == "Hello Lucky Luke !"
```

Un objet de type `str` (string, chaîne de caractères quoi) est *immutable*, on ne peut pas modifier
sa valeur.

```python
chaine = "pithon"
chaine[1] = 'y'
```

## Les structures de données


## Les listes


- Les listes sont des *sequences* (`str`, `tuple`, `list`)
- Les *sequences* sont des structures de données indicées qui peuvent contenir des éléments de différents types
- Les *sequences* sont des *iterables*, les listes aussi donc
- Les éléments d'une liste peuvent être modifiés (*mutable*)
- On accède à un élément par son indice (de 0 à n-1, n étant le nombre d'éléments)

- Une liste vide peut se déclarer de deux façons

```python
stack = []
stack = list()
```

```python
stack = list("Pithon")
stack[1] = 'y'
stack
```

### ✍️ Exo 5 ✍️

```python
def change_char(string, index):
    """In the given string, change the char at given index for 'z' and return the modified str
    ex: change("maison", 2) -> mazson
    """
    pass # votre code ici
```

```python
assert isinstance(change_char("maison", 3), str)
assert change_char("maison", 3) == "maizon"
assert change_char("maison", 0) == "zaison"
```

## Les dictionnaires

- Un dictionnaire est une structure de données associative de type 'clé' → 'valeur'
- Les données ne sont pas ordonnées comme dans les listes
- On accède à une valeur par sa clé
- Les clés sont uniques : on ne peut pas associer deux valeurs à une même clé
- `keys()` renvoie la liste des clés, `values()` la liste des valeurs

```python
couleurs = {'a':'noir', 'e':'blanc', 'i':'rouge', 'u':'vert', 'o':'bleu'}
couleurs['i'] = "pourpre"
couleurs
```

```python
couleurs.keys()
```

```python
couleurs.values()
```

```python
couleurs.items()
```

# Les tuples

- Les tuples (`tuple`) sont des *séquences* similaires aux listes sauf qu'elles ne peuvent pas être
  modifiées (*immutable*)
- Les tuples sont souvent utilisées comme valeur de retour d'une fonction
- Les tuples peuvent être utilisées comme clé de dictionnaire

```python
voyelles = ('a', 'e', 'i', 'o', 'u', 'y')
my_var = tuple('Perl')
my_var
```

## Les structures conditionnelles

```python
if condition:
    [...]
elif condition:  # si besoin
    [...]
else:  # si besoin
    [...]
```

### Opérateurs booléens

``not`` négation  
``and`` conjonction (True si les deux opérandes sont vraies, False sinon)  
``or``  disjonction (True si une des deux opérandes est vraie)


- Les valeurs ci-dessous sont toutes évaluées par l'interpréteur comme ayant la valeur booléenne *false* :

  `False` `None` `0` (et les nombres qui lui sont égaux) `""` `()` `[]` `{}`

- Tout le reste<sup>1</sup> sera évalué comme _true_

  Vous pouvez écrire :
  ```python
  >>> if var: ou while my_list:  
  ```
  plutôt que :  

  ```python
  >>> if var != "": ou while my_list != []:
  ```

<sup>1</sup> <small>Sauf les objets dont vous avez construit les classes. Voir les diapos à venir sur Classes et objets.</small>
<!-- #endregion -->

```python
x = 4
if x > 3 and x <= 5:
    print("x a grandi, un peu")
elif x > 5:
    print("x a grandi")
else:
    print("x n'a pas grandi")
```

## Les boucles

- Les boucles `while` nécessitent que la valeur utilisée dans la condition d'arrêt soit modifiée
  dans le corps de la boucle.

```python
i = 1
while i < 5:
    print(i)
    i = i + 1
```

- Les boucles `for` s'appliquent sur les *séquences* (`list`, `str`, `tuple`) et plus généralement
  sur les *iterables* [voir doc](https://docs.python.org/3/glossary.html#term-iterable)
- Les *iterables* sont des objets issus de classes qui implémentent la méthode `__iter__()` et/ou
  `__getitem__()`
- L'instruction `continue` permet de passer à l'itération suivante
- L'instruction `break` permet de quitter la boucle en cours

```python
for item in voyelles:
    print(item)
```

```python
for item in couleurs.keys():
    if item == 'i':
        continue
    print(item)
```

```python
for key, value in couleurs.items():
    print(key, value)
    if key == 'i':
        break
```

- `zip` permet de boucler sur plusieurs séquences
- Si les séquences sont de tailles différentes `zip` s'arrête à la longueur la plus petite

```python
noms = ['einstein', 'planck', 'turing', 'curie', 'bohr', 'shannon']
facs = ['inalco', 'p3', 'p10', 'inalco', 'p3', 'inalco']
parcours = ['pro', 'r&d', 'r&d', 'pro', 'pro', 'r&d']
for nom, fac, parcours in zip(noms, facs, parcours):
    print(f"{nom} est inscrit en {parcours} à {fac}")
```

### ☕ Exos 6 ☕

```python
def fr_ar(string):
    """
    recherche les pronoms personnels dans la chaîne donnée en argument
    renvoie leurs équivalents en arabe sous forme de liste
    """
    res = []
    # from https://fr.wikipedia.org/wiki/Liste_Swadesh_de_l%27arabe
    fr_ar_dict = {'je':'أنا', 'tu':'أنت', 'il': 'هو', 'nous': 'نحن', 'vous': 'انتما', 'ils': 'هما'}
    # votre code ici
```

```python
assert fr_ar("trop bizarre cet exercice") == []
assert fr_ar("il nous a rien dit") == ['هو', 'نحن']
```

#### 1. Des triangles

1. Écrire une fonction `la_plus_grande(longueur1, longueur2, longueur3)` qui renvoie la longueur du
   plus grand côté (une fonction de python fait peut-être déjà cela...).
2. Écrire une fonction `est_equilateral(longueur1, longueur2, longueur3)` qui détermine si un
   triangle est équilatéral ou non (les trois côtés ont la même longueur).
3. Écrire une fonction `est_isocele(longueur1, longueur2, longueur3)` qui détermine si un triangle
   est isocèle (deux côtés de même longueur mais pas trois) ou non.
4. Écrire une fonction `caracteristiques(longueur1, longueur2, longueur3)` qui renvoie la nature et
   la taille du plus grand côté d'un triangle. On dira qu'un triangle est `quelconque` s'il n'est ni
   équilatéral ni isocèle. Affiche `pas un triangle` si les longueurs données ne font pas un
   triangle (la longueur du plus grand côté est supérieure à celle des deux autres).

```python
def la_plus_grande(longueur1, longueur2, longueur3):
    """Renvoie la plus grande longueur."""
    pass # TODO: codez !

def est_equilateral(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est équilatéral."""
    pass # TODO: codez !

def est_isocele(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est isocele."""
    pass # TODO: codez !

def est_triangle(longueur1, longueur2, longueur3):
    """Renvoie si les longueurs données font bien un triangle."""
    pass # TODO: codez !

def caracteristiques(longueur1, longueur2, longueur3):
    """Renvoie les caractéristiques d'un triangle.
    Les caractéristiques d'un triangle sont :
        - sa nature
        - la taille de son plus grand côté.

    On dira qu'un triangle est `quelconque` s'il n'est ni équilatéral ni isocèle.

    Affiche `pas un triangle` si les longueurs données ne font pas un triangle
    (la longueur du plus grand côté est supérieure à celle des deux autres).
    """
    pass # TODO: codez !
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

#### 2. Des heures

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
    pass # TODO: codez !

def secondes(heure):
    """Prend une heure au format `H:M:S` et renvoie le nombre de secondes
    correspondantes (entier).

    On suppose que l'heure est bien formattée. On aura toujours un nombre
    d'heures valide, un nombre de minutes valide et un nombre de secondes valide.
    """
    pass # TODO: codez !
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

3. Des cartes

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
    """Renvoie la carte qui remporte le pli en faisant attention aux couleurs :
        - la carte du premier joueur `carte1` donne la couleur attendue.
        - une carte qui n'est pas à la bonne couleur perd automatiquement.

    On ne gèrera pas certains cas incohérents comme une carte ou un pli invalide.
    """
    pass # TODO: codez !
```

```python
assert(gagne_couleur('S1', 'S2', 'S3', 'S4')) == 'S4'
assert(gagne_couleur('S4', 'S3', 'S2', 'S1')) == 'S4'
assert(gagne_couleur('S1', 'D2', 'C3', 'H4')) == 'S1'
assert(gagne_couleur('S1', 'D2', 'S13', 'S10')) == 'S13'
```

Faire l'exercice ASCII ART de codingame: <https://www.codingame.com/ide/puzzle/ascii-art>
