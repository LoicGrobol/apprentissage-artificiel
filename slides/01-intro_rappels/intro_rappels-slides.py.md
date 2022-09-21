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

<!-- #region slideshow={"slide_type": "slide"} -->
<!-- LTeX: language=fr -->

Cours 1 : Introduction et rappels Python
========================================

**Loïc Grobol** [\<lgrobol@parisnanterre.fr\>](mailto:lgrobol@parisnanterre.fr)

2021-09-22

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Bonjour

- Loïc Grobol (il/iel) [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)
- PHILLIA / MoDyCo (Bâtiment Rémond, 4ème, bureau 404C)
- *Office hours* le mardi après-midi, n'hésitez pas à passer y compris sans rendez-vous (mais je
  préfère si vous m'envoyez un mail pour me prévenir)
- De manière générale, n'hésitez pas à m'écrire

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Infos pratiques

- **Quoi** « Apprentissage artificiel »
- **Où** Salle 219, bâtiment Paul Ricœur
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 20/09 au 17/11
  - Voir le planning pour les dates exactes (quand il aura été mis en ligne)

→ PC portable obligatoire pour les cours, de préférence chargé. Si c'est un problème parlez m'en
tout de suite et on trouvera une solution.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Liens

- La page du cours (slides, documents, nouvelles, consignes…)
  - → <https://loicgrobol.github.io/apprentissage-artificiel>
- Le dépôt GitHub (sources, compléments et historique)
  - → <https://github.com/LoicGrobol/apprentissage-artificiel>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Setup

- Tous les notebooks s'ouvrent dans [Binder](https://mybinder.org), y compris les slides
- Pour l'instant on peut s'en contenter, pour la suite ça ne suffira pas
- Pour la séance prochaine, il faudra avoir Python 3 installé (mais c'est déjà votre cas à toustes,
  non ?)

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Aujourd'hui

Crash course Python
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print("C'est parti")
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Opérateurs mathématiques et fonctions

### Opérateurs

- `+` addition
- `-` soustraction
- `*` multiplication
- `/` division
- `//` la division entière
- `%` modulo (reste de la division)
- `**` puissance

- L'ordre des opérations est l'ordre classique en mathématiques (puissance passe avant les
  opérations).
- On peut utiliser des parenthèses pour définir des priorités.
- Lire [la doc](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)

<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
2707+3*2
```

```python slideshow={"slide_type": "fragment"}
(2707+3)*2
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Les fonctions

Vous connaissez les fonctions
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print(2713)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Ici on a fait **un appel** de la fonction `print` avec comme **argument** le nombre `2713`.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Il y a plein de fonctions déjà définies
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
len("abcde")
```

```python slideshow={"slide_type": "fragment"}
bool(5)
```

```python slideshow={"slide_type": "fragment"}
abs(-12)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
…
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Et on peut en définir des nouvelles
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
def ma_super_fonction(un_argument):
    print(un_argument)
    print(un_argument)
```

```python slideshow={"slide_type": "fragment"}
ma_super_fonction(15)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Si on veut que la fonction donne un résultat, comme `abs` par exemple, on le fait avec `return`
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
def ma_nouvelle_fonction(nombre):
    return nombre + 2
```

```python slideshow={"slide_type": "fragment"}
ma_nouvelle_fonction(2715)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
**Attention**: *afficher* un résultat et *renvoyer* un résultat ce n'est pas la même chose
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
def mon_incroyable_affichage(nombre):
    print(nombre + 3)

def mon_incroyable_retour(nombre):
    return nombre + 2
```

```python slideshow={"slide_type": "fragment"}
mon_incroyable_retour(27) 
```

```python slideshow={"slide_type": "subslide"}
mon_incroyable_affichage(27)
```

```python slideshow={"slide_type": "fragment"}
mon_incroyable_retour(27) + 3
```

```python slideshow={"slide_type": "fragment"} tags=["raises-exception"]
mon_incroyable_affichage(27) + 3
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exos 1 ✍️

C'est à vous de jouer !

Vous avez une fonction à compléter (ça vous apprendra à écrire des fonctions 😤).  

À chaque fois j'essaierai d'ajouter une cellule avec des tests qui vous permettront de valider votre
code. Écrivez votre code dans la cellule de la fonction (et enlevez `pass`), exécutez cette cellule
(bouton 'Run' ou ctrl + Enter) puis exécutez la cellule de test.

L'objectif est que vous soyez autonome pour valider ces exos (et accessoirement de vous familiariser
avec les tests).
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
def square(num):
    """Renvoie le nombre donné en argument au carré."""
    pass # Votre code ici
```

```python slideshow={"slide_type": "-"}
assert square(3) == 9
assert square(0) == 0
assert square(-2) == 4
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Opérateurs de comparaison

- `<` inférieur  / `<=` inférieur ou égal
- `>` supérieur  / `>=` supérieur ou égal
- `==` égal / `!=` différent
- `is` identité (pour les objets surtout)/ `is not` non identité

Lire [la doc](https://docs.python.org/3/library/stdtypes.html#comparisons).
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 2 ✍️
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
def on_fait_la_taille(moi, toi):
    """Vrai ssi `moi` est plus grand que `toi`"""
    pass # Votre code ici
```

```python slideshow={"slide_type": "-"}
assert on_fait_la_taille(100, 80) == "plus grand"
assert on_fait_la_taille(100, 120) == "plus petit"
assert on_fait_la_taille(100, 100) == "pareil"
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Identité et égalité

`a == b` est vrai si `a` et `b` sont égaux, `a is b` si c'est le même objet.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
display(["spam"] == ["spam"])
display(["spam"] is ["spam"])
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Il y a quelques pièges, mais on y reviendra
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Les variables

- L'affectation des variables se fait à l'aide du symbole `=`  
- Si la variable est placée à droite du symbole `=`, sa *valeur* est affectée à la variable placée à
  gauche.
- Les noms de variable sont composés de caractères alphabétiques (min ou maj), des chiffres et de
  l'underscore.
- Les noms de variable sont choisis par le programmeur, ils doivent être le plus clair possible. Il
  est conseillé de suivre la [PEP 8](https://www.python.org/dev/peps/pep-0008/).
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
spam = 3 + 2
print(spam)

eggs = spam
print(eggs)
```

```python slideshow={"slide_type": "fragment"}
je-ne-suis-pas-une-variable = 2 
```

```python slideshow={"slide_type": "fragment"}
3_moi_non_plus = 2 + 3
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- À part ça, seuls les mots réservés sont interdits.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import keyword
print(keyword.kwlist)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On *peut* faire des trucs exotiques (voir la [doc](https://docs.python.org/3/reference/lexical_analysis.html#identifiers))
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
z̷̫̻̘̞̫͓̩̮͐̄̀̇̐̅̈́̂̊͂̚͜͝͝ā̷̧̛̻͎͙̣̻̫̹̙̠̖̬̏̈́͒͋̃́̄̿͋͛̊l̶̢̡̹̠̜͉̗̆̎̐̀͑͗̒̒́́̅̿͒͜g̶̢̡̼̭̭̫̽̄̓̌͗͠͝ó̴͇̯͚̮̟̻͕̭͂͑̅͐̿͂͗͌͌̌̓ͅ = "HE COMES"
print(z̷̫̻̘̞̫͓̩̮͐̄̀̇̐̅̈́̂̊͂̚͜͝͝ā̷̧̛̻͎͙̣̻̫̹̙̠̖̬̏̈́͒͋̃́̄̿͋͛̊l̶̢̡̹̠̜͉̗̆̎̐̀͑͗̒̒́́̅̿͒͜g̶̢̡̼̭̭̫̽̄̓̌͗͠͝ó̴͇̯͚̮̟̻͕̭͂͑̅͐̿͂͗͌͌̌̓ͅ)
```

<!-- #region slideshow={"slide_type": "fragment"} -->
MAIS ON NE LE FAIT PAS
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Les types

- Python est un langage à typage *dynamique* fort : le type d'une variable est déterminé par
  l'interpréteur.
- Python est un langage à typage dynamique *fort* : pas de conversion implicite, certaines actions
  sont interdites.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
"Hello" + 1
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- La fonction `type()` retourne le type de la variable donnée en argument.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
type("Hello")

```

<!-- #region slideshow={"slide_type": "subslide"} -->
- La fonction `isinstance(obj, class)` vous dit si l'objet donné en argument est de la classe
  'class' ou non
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
isinstance('hello', int)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 3 ✍️

Vous reprenez votre fonction `square` de façon à afficher "Erreur de type" quand l'argument n'est pas de type `int`
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
def square(num):
    """Renvoie le nombre donné en argument au carré."""
    pass # Votre code ici
```

```python slideshow={"slide_type": "-"}
assert square(3) == 9
assert square(0) == 0
assert square(-2) == 4
square("test")
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les chaînes de caractère
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
- Les chaînes de caractères sont entourées de quotes simples `'` ou doubles `"`
- Si votre mot contient une apostrophe, entourez-le de guillemets `"`
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
'Ça donne une erreur t'as vu'
```

```python slideshow={"slide_type": "fragment"}
"Ça donne une erreur t'as vu"
```

<!-- #region slideshow={"slide_type": "subslide"} -->
On peut aussi utiliser trois quotes pour avoir une chaîne de caractères sur plusieurs lignes
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
"""Ceci est une
chaîne de caractères
sur plusieurs lignes
Je peux y mettre des simples ' et double " quotes sans problème !
"""
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Les chaînes sont des **séquences de caractères**, on peut leur appliquer les opérations suivantes
propres à la catégorie d'objets *séquences* :

(Vous connaissez d'autres *séquences* au fait ?)

- longueur, minimum, maximum
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
spam = "bonjour"
print(len(spam))
print(max(spam))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- *index*
  - Les indices commencent à `0` !
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
spam = "bonjour"
print(spam[2])
print(spam[-1])
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- *slice*
  - `spam[i:j]`, c'est `spam[i]`, `spam[i+1]`, …, `spam[j-1]`
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
print(spam[0:3]) # 3 premiers éléments
print(spam[-3:]) # 3 derniers éléments
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- *appartenance*
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
if 'u' in spam:
    print("Il y a un u dans {}".format(spam))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Les chaînes ont aussi des fonctions qui leur sont propres

Voir [la liste complète dans la
doc](https://docs.python.org/3/library/stdtypes.html#string-methods)

- `lower()` transforme la chaine en minuscules
- `upper()` transforme la chaine en majuscules
- `replace(old, new)` remplace les occurrences de `old` par `new`
- `strip(chars=None)` appelé sans arguments supprime le ou les espaces en tête et en fin de chaîne  
- `rstrip(chars=None)` fait la même chose en fin de chaîne uniquement
- `lstrip(chars=None)` idem en début de chaîne
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
- `split(sep=None)` découpe une chaîne en fonction de `sep` et renvoie une liste. Si `sep` n'est pas
  donné, coupe sur tous les caractères d'espace
- `join(iterable)` est l'inverse de `split`, il permet de joindre les éléments d'un *iterable* pour
  former une seule chaîne de caractères
  [`format()`](https://docs.python.org/3/library/string.html#formatstrings) pour effectuer
  l'[interpolation de chaîne](https://en.wikipedia.org/wiki/String_interpolation)
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
words = "bonjour ça va ?".split(' ')
"-".join(words)
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Formatage de chaînes

> There should be one — and preferably only one — obvious way to do it.  ([PEP 20 : *Zen of
> Python*](https://www.python.org/dev/peps/pep-0020/))

Sauf que :

- Concaténation avec `+` [comme n'importe quelle
  séquence](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations) **à éviter**
- [f-string](https://docs.python.org/3/library/string.html#formatstrings)
- Interpolation avec [`format()`](https://docs.python.org/3/library/functions.html#format)
- Et encore d'autres dont on ne parlera pas.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
name = "Morgan"
coffee_price = 0.6

print("Tiens salut " + name + ". T'aurais pas " + str(coffee_price*2) + " euros pour 2 cafés ?")

print(f"Tiens salut {name}. T'aurais pas {coffee_price*2} euros pour 2 cafés ?")

print("Tiens salut {}. T'aurais pas {} euros pour 2 cafés ?".format(name, coffee_price*2))
```

**Si possible utiliser des *f-strings*** (c'est presque toujours possible).

<!-- #region slideshow={"slide_type": "subslide"} -->
On évite de faire ça avec `+` parce que c'est moins lisible et que c'est **lent**. De fait on créée
une chaîne intermédiaire à chaque étape.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
s1 = "Tiens salut " + name
s2 = s1 + ". T'aurais pas "
s3 = s2 + str(coffee_price*2)
s4 = s3 + " euros pour 2 cafés ?"
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Sur l'exemple ci-dessus ça va, mais on se retrouve vite à additionner des centaines de chaînes et
c'est la galère.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Pour concaténer beaucoup de chaînes il vaut mieux utiliser `join`
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print(" 👏 ".join(["On", "ne", "concatène", "pas", "des", "chaînes", "de", "caractères", "avec", "+"]))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Attention aussi à la concaténation implicite
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
spam = ("Hello, " "there")
ham = ("General ", "Kenobi")
print(spam)
print(ham)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 4 ✍️
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
def say_hello(firstname, lastname):
    # avec des f-string svp
    # votre code ici
    pass
```

```python slideshow={"slide_type": "-"}
assert say_hello("Lucky", "Luke") == "Hello Lucky Luke !"
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Un objet de type `str` (string, chaîne de caractères quoi) est *immutable*, on ne peut pas modifier
sa valeur.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
chaine = "pithon"
chaine[1] = 'y'
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les structures de données

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Les listes
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
- Les listes sont des *sequences* (comme `str`, `tuple`, `list`)
- Les *sequences* sont des structures de données indicées qui peuvent contenir des éléments de
  différents types
- Les *sequences* sont des *iterables*, les listes aussi donc
- Les éléments d'une liste peuvent être modifiés (*mutable*)
- On accède à un élément par son indice (de 0 à n-1, n étant le nombre d'éléments)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Une liste vide peut se déclarer de deux façons
<!-- #endregion -->

```python
stack = []
stack = list()
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Les listes, elles, sont *mutables*
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
stack = list("Pithon")
stack[1] = 'y'
stack
```

<!-- #region slideshow={"slide_type": "fragment"} -->
C'est même le prototype d'une séquence mutable, elles servent à tout, partout, en Python.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 5 ✍️
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
def change_char(s, idx):
    """In the given string, change the char at given index for 'z' and return the modified str
    ex: change("maison", 2) -> mazson
    """
    pass # votre code ici
```

```python slideshow={"slide_type": "-"}
assert isinstance(change_char("maison", 3), str)
assert change_char("maison", 3) == "maizon"
assert change_char("maison", 0) == "zaison"
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Les dictionnaires

- Un dictionnaire est une structure de données associative de type 'clé' → 'valeur'
- Les données ne sont pas ordonnées comme dans les listes
- On accède à une valeur par sa clé
- Les clés sont uniques : on ne peut pas associer deux valeurs à une même clé
- `keys()` renvoie la liste des clés, `values()` la liste des valeurs
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
couleurs = {'a':'noir', 'e':'blanc', 'i':'rouge', 'u':'vert', 'o':'bleu'}
couleurs['i'] = "pourpre"
couleurs
```

```python slideshow={"slide_type": "-"}
couleurs.keys()
```

```python slideshow={"slide_type": "-"}
couleurs.values()
```

```python slideshow={"slide_type": "-"}
couleurs.items()
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Les tuples

- Les tuples (`tuple`) sont des *séquences* similaires aux listes sauf qu'elles ne peuvent pas être
  modifiées (*immutable*).
- Les tuples sont souvent utilisées comme valeur de retour d'une fonction.
- Contrairement aux listes, les tuples peuvent être utilisés comme clé de dictionnaire, à votre
  avis pourquoi ?.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
voyelles = ('a', 'e', 'i', 'o', 'u', 'y')
my_var = tuple('Perl')
my_var
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les structures conditionnelles
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
```python
if condition:
    [...]
elif condition:  # si besoin
    [...]
else:  # si besoin
    [...]
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### Opérateurs booléens

- `not` négation  
- `and` conjonction (`True` si les deux opérandes sont vraies, `False` sinon)  
- `or` disjonction (`True` si une des deux opérandes est vraie)
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
- Les valeurs ci-dessous sont toutes évaluées par l'interpréteur comme ayant la valeur booléenne
  `False` :

  `False` `None` `0` (et les nombres qui lui sont égaux) `""` `()` `[]` `{}`

- Tout le reste<sup>1</sup> sera évalué comme `True`

  Vous pouvez écrire `if var` ou `while my_list` plutôt que `if var != ""` ou `while my_list != []`

<sup>1</sup> <small>Sauf les objets dont vous avez construit les classes. Voir les diapos à venir
sur Classes et objets.</small>
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
x = 4
if x > 3 and x <= 5:
    print("x a grandi, un peu")
elif x > 5:
    print("x a grandi")
else:
    print("x n'a pas grandi")
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les boucles

- Les boucles `while` nécessitent que la valeur utilisée dans la condition d'arrêt soit modifiée
  dans le corps de la boucle.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
i = 1
while i < 5:
    print(i)
    i = i + 1
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- Les boucles `for` s'appliquent sur les *séquences* (`list`, `str`, `tuple`) et plus généralement
  sur les *iterables* [voir doc](https://docs.python.org/3/glossary.html#term-iterable)
- Les *iterables* sont des objets issus de classes qui implémentent la méthode `__iter__()` et/ou
  `__getitem__()`
- L'instruction `continue` permet de passer à l'itération suivante
- L'instruction `break` permet de quitter la boucle en cours
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
for item in "aeiouy":
    print(item)
```

```python slideshow={"slide_type": "subslide"}
for item in couleurs:
    if item == 'i':
        continue
    print(item)
```

```python slideshow={"slide_type": "subslide"}
for key, value in couleurs.items():
    print(key, value)
    if key == 'i':
        break
```

- `zip` permet de boucler sur plusieurs séquences
- Si les séquences sont de tailles différentes `zip` s'arrête à la longueur la plus petite

```python slideshow={"slide_type": "subslide"}
noms = ['einstein', 'planck', 'turing', 'curie', 'bohr', 'shannon']
facs = ['inalco', 'p3', 'p10', 'inalco', 'p3', 'inalco']
parcours = ['pro', 'r&d', 'r&d', 'pro', 'pro', 'r&d']
for nom, fac, parcours in zip(noms, facs, parcours):
    print(f"{nom} est inscrit en {parcours} à {fac}")
```

<!-- #region slideshow={"slide_type": "slide"} -->
### ☕ Exos 6 ☕
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
#### 1. Des triangles

1. Écrire une fonction `la_plus_grande(longueur1, longueur2, longueur3)` qui renvoie la longueur du
   plus grand côté (une fonction de python fait peut-être déjà cela...).
2. Écrire une fonction `est_equilateral(longueur1, longueur2, longueur3)` qui détermine si un
   triangle est équilatéral ou non (les trois côtés ont la même longueur).
3. Écrire une fonction `est_isocele(longueur1, longueur2, longueur3)` qui détermine si un triangle
   est isocèle (deux côtés de même longueur, mais pas trois) ou non.
4. Écrire une fonction `caracteristiques(longueur1, longueur2, longueur3)` qui renvoie la nature et
   la taille du plus grand côté d'un triangle. On dira qu'un triangle est `quelconque` s'il n'est ni
   équilatéral ni isocèle. Affiche `pas un triangle` si les longueurs données ne font pas un
   triangle (la longueur du plus grand côté est supérieure à celle des deux autres). On peut
   commencer par écrire la fonction `est_triangle` pour vérifier cette dernière condition.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
def la_plus_grande(longueur1, longueur2, longueur3):
    """Renvoie la plus grande longueur."""
    pass # TODO: codez !

def est_equilateral(longueur1, longueur2, longueur3):
    """Renvoie True si un triangle est équilatéral, False sinon."""
    pass # TODO: codez !

def est_isocele(longueur1, longueur2, longueur3):
    """Renvoie True si un triangle est isocele, False sinon."""
    pass # TODO: codez !

def est_triangle(longueur1, longueur2, longueur3):
    """Renvoie True si les longueurs données font bien un triangle, False sinon."""
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

```python slideshow={"slide_type": "subslide"}
assert caracteristiques(1, 1, 1) ==  ("equilatéral", 1)
assert caracteristiques(1, 1, 2) == ("isocèle", 2)
assert caracteristiques(1, 2, 1) == ("isocèle", 2)
assert caracteristiques(2, 1, 1) == ("isocèle", 2)
assert caracteristiques(2, 3, 1) == ("quelconque", 3)
assert caracteristiques(2, 3, 6) == "pas un triangle"
assert caracteristiques(6, 3, 2) == "pas un triangle"
assert caracteristiques(2, 6, 3) == "pas un triangle"
```

<!-- #region slideshow={"slide_type": "subslide"} -->

#### 2. La parité

<!-- #endregion -->

```python
def is_even(num):
    """
    Renvoie True si num est pair, False sinon
    """
    # votre code ici
```

```python
assert is_even(1) == False
assert is_even(2) == True
assert is_even(-3) == False
assert is_even(-42) == True
assert is_even(0) == True
```

<!-- #region slideshow={"slide_type": "subslide"} -->
#### 3. Des heures

1. Écrire une fonction `heures(secondes)` qui prend un nombre de secondes (entier) et le convertit
   en heures, minutes et secondes sous le format `H:M:S` où `H` est le nombre d'heures, `M` le
   nombre de minutes et `S` le nombre de secondes.
2. Écrire une fonction `secondes(heure)` qui prend une heure au format `H:M:S` et renvoie le nombre
   de secondes correspondantes (entier).

On ne gèrera ici pas les cas incohérents comme un nombre de secondes négatif ou une heure mal
formatée.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
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

```python slideshow={"slide_type": "subslide"}
assert (heures(0)) == "0:0:0"
assert(heures(30)) == "0:0:30"
assert(heures(60)) == "0:1:0"
assert(heures(66)) == "0:1:6"
assert(heures(3600)) == "1:0:0"
assert(heures(86466)) == "24:1:6"
assert(secondes('0:0:0')) == 0
assert(secondes('6:6:6')) == 21966
assert(secondes(heures(86466))) == 86466
assert(heures(secondes("24:1:1"))) == "24:1:1"
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les listes : fonctions

Les listes héritent des fonctions des *sequences*, elles ont également des [méthodes
propres](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
stack = [12, 15, 12, 7, 18]
```

Parmi ces fonctions, nous utiliserons principalement :

<!-- #region slideshow={"slide_type": "subslide"} -->
- `append(x)` : ajoute un élément `x` à la fin de la liste (haut de la pile*)
<!-- #endregion -->

```python
stack.append(3)
display(stack)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `extend(lst)` : ajoute tous les éléments de `lst` à la fin de la liste
<!-- #endregion -->

```python
stack.extend([10, 11])
display(stack)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `pop(index=-1)` : supprime et renvoie l'élément de la liste à la position `index`
<!-- #endregion -->

```python
h = stack.pop()
display(h)
display(stack)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `index(x)` : renvoie l'index du premier élément de valeur x
<!-- #endregion -->

```python
stack.index(12)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `count(x)` : renvoie le nombre de fois où x apparaît

  **Attention** : si vous avez plusieurs éléments à compter, utilisez plutôt
  [`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter)
<!-- #endregion -->

```python
stack.count(12)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `sort(key=None, reverse=False)` : trie et modifie la liste, lire la
    [doc](https://docs.python.org/3/howto/sorting.html#sortinghowto) pour en savoir plus sur les
    ordres de tri.
<!-- #endregion -->

```python
stack.sort()
display(stack)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Attention à ne pas confondre `append` et `extend`
<!-- #endregion -->

```python
stack.append(23)
display(stack)
```

```python
stack.append([35, 46])
display(stack)
```

```python
stack.extend([51, 52])
display(stack)
```

<!-- #region slideshow={"slide_type": "slide"} -->
### ✍️ Exo 7 ✍️
<!-- #endregion -->

```python
def tokenize(sentence):
    """
    Tokenize la phrase donnée en argument (sep = espace).
    Renvoie une liste de mots. Les mots composés avec un tiret
    sont décomposés dans des sous-listes.
    Args:
        sentence (string): la phrase à tokenizer
    Returns:
        list
    """
    pass # À vous
```

```python
assert tokenize("je suis né dans le gris par accident") == \
    ['je', 'suis', 'né', 'dans', 'le', 'gris', 'par', 'accident']
assert tokenize("tout mon cœur est resté là-bas") == \
    ['tout', 'mon', 'cœur', 'est', 'resté', ['là', 'bas']]
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Les listes en compréhension

- Elles permettent de définir des listes par filtrage ou opération sur les éléments d'une autre
  liste
- La [PEP 202](http://www.python.org/dev/peps/pep-0202/) conseille de préférer les listes en
  compréhension aux fonctions `map()` et `filter()`  
- C'est puissant et concis, *so pythonic*
<!-- #endregion -->

```python
[i ** 2 for i in range(10)]
```

```python slideshow={"slide_type": "subslide"}
[i ** 2 for i in range(10) if i % 2 == 0]
```

```python slideshow={"slide_type": "subslide"}
[(i, j) for i in range(2) for j in ['a', 'b']]
```

<!-- #region slideshow={"slide_type": "slide"} -->
### ✍️ Exo 8 ✍️

Utilisez une liste en compréhension sur la sortie de votre fonction `tokenize` de manière à ne
retenir que les noms composés
<!-- #endregion -->

```python
words = tokenize("De-ci de-là, cahin-caha, va trottine, va chemine, va petit âne")
compounds = [] # ← modifiez ça
assert compounds == [['De', 'ci'], ['de', 'là,'], ['cahin', 'caha,']]
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Parcours de liste

La boucle `for` est particulièrement adaptée pour parcourir les itérables et donc les listes
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
voyelles = ['a', 'e', 'i', 'o', 'u']
for item in voyelles:
    print(item)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
La fonction `enumerate` peut être utile dans certains cas, elle renvoie un `tuple` contenant
l'indice et la valeur de l'item à l'indice concerné
<!-- #endregion -->

```python
for i, item in enumerate(voyelles):
    print(i, item)
```

C'est de très loin préférable à itérer sur `range(len(voyelles))`.

<!-- #region slideshow={"slide_type": "slide"} -->
### Copie

Dans `y = x`, `y` n'est pas une copie de x, les deux pointent vers le même objet. C'st
particulièrement important pour les objets *mutables* comme les listes.
<!-- #endregion -->

```python
x = [1, 2, 3]
y = x
y[0] = 4
```

```python
display(x)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Si ce qu'on veut copier est une liste, on peut utiliser
<!-- #endregion -->

```python
x = [1, 2, 3]
y = x[:]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
ou
<!-- #endregion -->

```python
y = list(x)
y[0] = 4
x
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Il y a d'autres façons de faire. Pour les objets complexes on peut regarder du côté du module
[`copy`](https://docs.python.org/3/library/copy.html), mais il n'y a pas de réponse universelle et
copier c'est souvent coûteux. Le mieux à faire quand on a envie de faire une copie, c'est de
commencer par se demander si on en a vraiment besoin.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Déballage de séquences

- Le *sequence unpacking* permet d'effectuer plusieurs affectations simultanées
- L'*unpacking* s'applique souvent sur des tuples
<!-- #endregion -->

```python
x, y, z = (1, 2, 3)
y
```

```python
lexique = [("maison", "mEz§"), ("serpent", "sERp@")]
for ortho, phon in lexique:
    print(phon)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- On peut aussi utiliser `*` pour déballer une séquence en argument de fonction
<!-- #endregion -->

```python
bornes = (0, 10)
for i in range(*bornes):
    print(i)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les ensembles

Les ensembles ([`set`](https://docs.python.org/3/library/stdtypes.html#set)) sont des collections
non ordonnées d'éléments sans doublons Les ensembles supportent les fonctions mathématiques d'union,
d'intersection, de différence :

- `value in s` renvoie si `value` est un élément de `s`
- `union(*sets)` renvoie l'union de tous les `sets` (l'ensemble des valeurs contenues dans tous les
  sets).
- `intersection(*sets)` renvoie l'intersection de tous les `sets` (l'ensemble des valeurs contenues
  dans au moins un set).

<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
ens0 = set()  # on crée l'ensemble vide
ens0
```

```python
ens1 = {'le', 'guépard', 'le', 'poursuit'}
ens1
```

```python
ens2 = {"avec", "le", "chandelier", "dans", "la", "cuisine"}
ens1.intersection(ens2)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
Attention il y a un piège
<!-- #endregion -->

```python
a = {1}
type(a)
```

```python
b = {}
type(b)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## ✍️ Exo 9

Dans cet extrait de données tirées des [listes de Swadesh de langues
austronésiennes](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists), ici pour le
tagalog et le cebuano, trouvez les mots en commun.
<!-- #endregion -->

```python
tagalog = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'tayo', 'you_pl':'kayo', 'they':'sila',\
           'this':'ito', 'that':'iyan', 'here':'dito', 'there':'doon', 'who':'sino',\
           'what':'ano', 'where':'saan', 'when':'kailan', 'how':'paano'}
cebuano = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'kita', 'you_pl':'kamo', 'they':'sila',\
           'this':'kiri', 'that':'kana', 'here':'diri', 'there':'diha', 'who':'kinsa',\
           'what':'unsa', 'where':'asa', 'when':'kanus-a', 'how':'unsaon'}
# Votre code ici
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Les dictionnaires : suite

- Les dictionnaires ([`dict`](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict))
  sont des structures de données associatives de type clé: valeur.
- Les clés d'un dictionnaire sont uniques, seuls les types *hashable* (*immutable* et objets que
  vous avez définis) peuvent être des clés.

  - `key in d` renvoie `True` si `key` est une clé de `d`
  - `keys()` renvoie la liste des clés
  - `values()` renvoie la liste des valeurs
  - `items()` renvoie la liste des couples clé:valeur (tuple)
  - `get(key, default=None)` renvoie la valeur associée à `key`. Si `key` n'existe pas, renvoie
    l'argument `default`. Ne modifie pas le dictionnaire.
  - `setdefault(key, default=None)` si `key` n'existe pas, insère `key` avec la valeur `default`
    dans le dictionnaire puis renvoie la valeur associée à la clé.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
d = {'Perl':'Larry Wall', 'Python':'Guido Van Rossum', 'C++':'Bjarne Stroustrup'}
d['Perl']
```

```python
d['Ruby']
```

```python
d.setdefault('Ruby', 'je sais pas')
d
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Module collections

- Le module *collections* propose des implémentations de structures de données supplémentaires
- Dans la liste (voir [doc](https://docs.python.org/3/library/collections.html)), deux pourront
  nous intéresser :

  - `defaultdict`

`defauldict` est similaire à un `dict` mais il permet l'autovivification

Son implémentation le rend plus rapide qu'un dictionnaire utilisé avec la fonction `setdefault`

<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
import collections
lexique = [("couvent", "kuv"), ("couvent", "kuv@")]
dico = collections.defaultdict(list)
for ortho, phon in lexique:
    dico[ortho].append(phon)
dico
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `Counter`
  
`Counter` est un dictionnaire où les valeurs attendues sont les nombres d'occurrences des clés
<!-- #endregion -->

```python
from collections import Counter
cnt = Counter()
lst = ['le', 'guépard', 'le', 'poursuit']
for item in lst:
    cnt[item] += 1
display(cnt)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 10

Faites la même chose avec un dictionnaire

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Les fichiers

- Pour travailler avec les fichiers on doit procéder à trois opérations :
   1. Ouverture avec la fonction [`open`](https://docs.python.org/3/library/functions.html#open)
      (lève l'exception `FileNotFoundError` en cas d'échec)
   2. Lecture (`read` ou `readline` ou `readlines`) et/ou écriture (`write`)
   3. Fermeture du fichier avec la fonction `close`
- Ouverture
  - `open` est une fonction qui accepte de nombreux arguments : lire [la
    doc](https://docs.python.org/3/library/functions.html#open)
  - `open` renvoie un objet de type `file`
  - Le plus souvent elle s'emploie de la manière suivante :

    ```python
      >>> #f = open(filename, mode)	   
      >>> f = open('nom_fichier', 'w')
    ```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
Les modes sont :

- `r` : lecture (défaut)
- `w` : écriture
- `x` : création et écriture (échec si le fichier existe déjà)
- `a` : concaténation (append)
- `b` : mode binaire
- `t` : mode texte (défaut)
- `+` : mise à jour

Voir [la doc](https://docs.python.org/3/library/functions.html#open) pour les détails
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## Les fichiers : ouverture

La documentation de Python conseille cette façon de faire :
<!-- #endregion -->
```python
with open('mon_fichier', 'r') as f:
    read_data = f.read()
```

L'utilisation du mot clé `with` garantit la fermeture du fichier même si une exception est soulevée.

<!-- #region slideshow={"slide_type": "subslide"} -->
## Les fichiers : lecture

- `read(size=-1)` lit les `size` premiers octets (mode `b`) ou caractères (mode `t`). Si `size` < 0,
  lit tout le fichier.
- `readline(size=-1)` lit au plus `size` caractères ou jusqu'à la fin de ligne. Si `size` < 0, lit
  toute la ligne. Il est conseillé de ne pas toucher à `size`.
- `readlines(hint=-1)` lit `hint` lignes du fichier. Si `hint` < 0, lit toutes les lignes du
  fichier.
- un objet `file` est un itérable ! (*the pythonic way*)
<!-- #endregion -->

```python
for line in f:
    process(line)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
## Les fichiers : écriture et fermeture

- `write(text)` écrit `texte` dans le fichier
- `close()` ferme le fichier.  

En règle générale veillez à toujours fermer les objets fichiers.  
En mode écriture oublier de fermer un fichier peut réserver des mauvaises surprises

- fonction `print`
<!-- #endregion -->
```python slideshow={"slide_type": "-"}
with open('mon_fichier', 'w') as output_f:
    for item in words:
        print(item, file=output_f)
```
<!-- #region slideshow={"slide_type": "subslide"} -->
- `sys.stdin`, `sys.stdout` et `sys.stderr` sont des objets de type `file`
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### ✍️ Exo 11

Lisez le fichier `data/austronesian_swadesh.csv` et écrivez les mots des langues Ilocano et Malagasy
dans deux fichiers distincts.  
Les données viennent de
[Wiktionary](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists).

(Essayez de faire comme si vous ne connaissiez pas le module csv sinon la partie qui suit n'aura
aucun intérêt.)
<!-- #endregion -->

```python
# c'est compliqué sans le module csv quand même
```

<!-- #region slideshow={"slide_type": "slide"} -->

## Module csv

La documentation est ici
: [https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html)  
Parce que les données au format csv sont très répandues et parce qu'il peut être pénible de le lire
correctement, le module csv est là pour vous aider.  
Pour le dire vite il y a deux façons de l'utiliser : reader/writer ou DictReader/DictWriter.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
- `csv.reader`
<!-- #endregion -->

```python
import csv

swadesh_light = []
with open('data/austronesian_swadesh.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"') # à l'ouverture je spécifie les séparateur de champ et de chaîne  
    for row in reader: # l'objet reader est un itérable
        swadesh_light.append(row[0:3])
        print(' | '.join(row[0:3])) # row est une liste de chaînes de caractères
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `csv.writer`
<!-- #endregion -->

```python
with open('swadesh_light.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='|',quotechar='"')
    #writer.writerows(swadesh_light) ici on écrit tout en une fois
    for item in swadesh_light:
        writer.writerow(item) # writerow reçoit une liste de chaînes
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- csv.DictReader  

Cette classe s'appuie sur la ligne d'en-tête pour créer une suite de dictionnaires.  
S'il n'y a pas de ligne d'en-tête on peut utiliser une liste `fieldnames` en paramètre.
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
with open('data/austronesian_swadesh.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',',quotechar='"')
    for row in reader: # ici row est un dictionnaire
         print(f"{row['Ilocano']} | {row['Malagasy']}")

```

<!-- #region slideshow={"slide_type": "subslide"} -->
- csv.DictWriter  

Cette fois il s'agit de générer un fichier csv à partir d'une séquence de dictionnaires. Le
paramètre `fieldnames` est obligatoire.
<!-- #endregion -->

```python
with open('swadesh_light.csv', 'w') as csvfile:
    fieldnames = ['english', 'ilocano']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|',quotechar='$')
    writer.writeheader()
    for num, en, ilo in swadesh_light:
        writer.writerow({'english': en, 'ilocano': ilo})

```

<!-- #region slideshow={"slide_type": "slide"} -->
## Module `re`
<!-- #endregion -->

```python
import re
```

<!-- #region slideshow={"slide_type": "subslide"} -->
- `re` est un module particulièrement important, vous devez lire la
  [doc](https://docs.python.org/3/library/re.html), absolument

- La doc officielle est parfois aride, ce [howto](https://docs.python.org/3/howto/regex.html)
  rédigé par A.M. Kuchling est plus digeste

a minima vous devez connaître les fonctions :

- `findall` : trouve toutes les occurences du motif, retourne une liste de chaînes trouvées
- `search` : trouve le motif, retourne un objet `Match`, `None` sinon
- `match` : détermine si le motif est présent au début de la chaîne, retourne un objet `Match`,
  `None` sinon
- `split` : découpe une chaîne selon un motif, retourne une liste de chaînes
- `sub` : remplace les occurences d'un motif par une chaîne de remplacement
- `compile` : compilation d'un motif (pattern), retourne un objet `Pattern`
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
if re.search(r"(\w|\s)+", "Un léopard me pourchasse"):
    print("Cours !")
```

```python
re.sub(r'e|é', 'i', 'éléphanteau')
```

<!-- #region slideshow={"slide_type": "subslide"} -->
## `\w` et Python3

`\w` est la classe prédéfinie des caractères alphanumériques :

- En Python 2 `\w` ~correspond~ correspondait à `[A-Za-z0-9_]`, avec les locales il est possible d'y
  ajouter d'autres caractères
- En Python 3 `\w` correspond à tous les caractères qui ont la propriété Unicode Letter d'après le
  module `unicodedata` (sauf si le motif est compilé en binaire ou si l'option `re.ASCII` est
  activée)
<!-- #endregion -->

```python
if re.search(r"\w", "馬青區團長成中央代表"):
    print("Yeah !")
```

```python
if re.search(r"\w", "هيلاري كلينتون"):
    print("Yeah !")
```

<!-- #region slideshow={"slide_type": "slide"} -->
### ☕ Exos 12 ☕
<!-- #endregion -->

1\. Écrire une fonction qui reçoit deux noms de langue austronésiennes, une liste de mots en anglais
et renvoie chacun des mots anglais avec leur traduction dans les deux langues.

```python
def get_austro_words(langue1, langue2, words):
    """
    Reçoit un couple de langues (langue1, langue2) et une liste de mots (words)
    Cherche dans la liste Swadesh des langues austronésiennes les traductions des mots
    dans ces deux langues.
    Renvoie un dictionnaire {'langue1': [w1, w2], 'langue2': [w1, w2]}
    Liste vide si la langue n'est pas répertoriée dans la liste
    """
    # votre code

    
assert get_austro_words('Malay', 'Malagasy', ['new', 'old', 'good']) == \
    {
        'Malay':['baharu', 'lama', 'bagus, baik'],
        'Malagasy':['vaovao', 'onta, hantitra', 'tsara']
    }
assert get_austro_words('Malay', 'Balinese', ['new', 'old', 'good']) == \
    {
        'Malay':['baharu', 'lama', 'bagus, baik'],
        'Balinese':[]
    }
```

2\. Pour chaque mot du Cebuano de la liste Swadesh austronésienne, trouvez les mots des autres
   langues qui ont les deux ou trois premiers caractères en commun.  
   (optionnel si vous voulez jouer avec les expressions régulières) Si le mot commence par une
   voyelle, elle pourra différer dans les autres langues. Ex: isa / usa seront considérées comme
   similaires (i/u) parce qu'à part la première lettre voyelle elles sont similaires.

3\. **Pour les champion⋅nes** Sans rechercher de solution sur internet, essayez d'implémenter une
   fonction qui calcule la distance de Levenshtein. (Vous pouvez chercher ce que c'est que la
   distance de Levenshtein et l'algorithme en pseudo-code, mais n'allez pas chercher directement
   d'implémentation en Python !)
