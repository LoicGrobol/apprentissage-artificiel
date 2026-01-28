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

2022-09-21

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

```python
assert on_fait_la_taille(100, 80) == "plus grand"
assert on_fait_la_taille(100, 120) == "plus petit"
assert on_fait_la_taille(100, 100) == "pareil"
```

## ✍️ Exo 3 ✍️

Vous reprenez votre fonction `square` de façon à afficher "Erreur de type" quand l'argument n'est
pas de type `int`

```python
def is_even(num):
    """Return True if `num` is even, False otherwise."""
    if not isinstance(num, int):
      return "Erreur de type"
    return num**2
```

```python
assert square(3) == 9
assert square(0) == 0
assert square(-2) == 4
square("test")
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
    """Renvoie True si un triangle est équilatéral, False sinon."""
    return longueur1 == longueur2 and longueur2 == longueur3

def est_isocele(longueur1, longueur2, longueur3):
    """Renvoie True si un triangle est isocele, False sinon."""
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

### 2. Parité

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

### 3. Des heures

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
assert(secondes('0:0:0')) == 0
assert(secondes('6:6:6')) == 21966
assert(secondes(heures(86466))) == 86466
assert(heures(secondes('24:1:1'))) == "24:1:1"
```

## ✍️ Exo 7 ✍️

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
    res = []
    for token in sentence.split():
        if "-" in token:
            res.append(token.split("-"))
        else:
            res.append(token)
    return res
```

On peut faire plus sophistiqué, mais ce n'est pas conseillé

```python
assert tokenize("je suis né dans le gris par accident") == \
    ['je', 'suis', 'né', 'dans', 'le', 'gris', 'par', 'accident']
assert tokenize("tout mon cœur est resté là-bas") == \
    ['tout', 'mon', 'cœur', 'est', 'resté', ['là', 'bas']]
```

### ✍️ Exo 8 ✍️

> Utilisez une liste en compréhension sur la sortie de votre fonction tokenize de manière à ne
> retenir que les noms composés

```python
words = tokenize("De-ci de-là, cahin-caha, va trottine, va chemine, va petit âne")
compounds = [w for w in words if not isinstance(w, str)]
assert compounds == [['De', 'ci'], ['de', 'là,'], ['cahin', 'caha,']]
```

Là encore on pourrait mieux faire, mais ça ne vaut pas vraiment le coup pour cet exo.

## ✍️ Exo 9

> Dans cet extrait de données tirées des [listes de Swadesh de langues
> austronésiennes](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists), ici pour le
> tagalog et le cebuano, trouvez les mots en commun.

```python
tagalog = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'tayo', 'you_pl':'kayo', 'they':'sila',\
           'this':'ito', 'that':'iyan', 'here':'dito', 'there':'doon', 'who':'sino',\
           'what':'ano', 'where':'saan', 'when':'kailan', 'how':'paano'}
cebuano = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'kita', 'you_pl':'kamo', 'they':'sila',\
           'this':'kiri', 'that':'kana', 'here':'diri', 'there':'diha', 'who':'kinsa',\
           'what':'unsa', 'where':'asa', 'when':'kanus-a', 'how':'unsaon'}
set(tagalog.values()).intersection(set(cebuano.values()))
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### ✍️ Exo 10


```python
from collections import Counter
cnt = Counter()
lst = ['le', 'guépard', 'le', 'poursuit']
for item in lst:
    cnt[item] += 1
display(cnt)
```

> Faites la même chose avec un dictionnaire

```python
lst = ['le', 'guépard', 'le', 'poursuit']
cnt = dict()
for item in lst:
    cnt[item] = cnt.get(item, 0) + 1
display(cnt)
```

### ✍️ Exo 11

> Lisez le fichier [`data/austronesian_swadesh.csv`](../../data/austronesian_swadesh.csv) et écrivez
> les mots des langues Ilocano et Malagasy dans deux fichiers distincts.
>
> Les données viennent de
> [Wiktionary](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists).
>
> (Essayez de faire comme si vous ne connaissiez pas le module csv sinon la partie qui suit n'aura >
> aucun intérêt.)

Pour commencer, ouvrez [`data/austronesian_swadesh.csv`](../../data/austronesian_swadesh.csv) avec
un éditeur de texte pour voir les problèmes :

- Il y a des sauts de lignes en plein milieu des cellules
- Il y a des cellules qui contiennent des virgules (qui est aussi le séparateur de colonnes).

```python
def read_that_ugly_csv(p):
    """Lit le fichier csv pourri en recollant les sauts de lignes intempestifs."""
    lines = []
    with open(p) as in_stream:
        next(in_stream)  # On saute la ligne d'en-tête
        for l in in_stream:
            # On saute les lignes vides ou blanches (en pratique la dernière)
            if not l or l.isspace():
                continue
            if l[0].isdigit(): # Débuts de lignes
                lines.append(l.strip())
            else: # fins de lignes tronquées : on les ajoute à la ligne d'avant
                lines.append(lines.pop()+l.strip())
    return lines

def get_malagasy_ilocano(lst):
    mal = []
    ilo = []
    for line in lst:
        # Dégage la colonne 0
        row = line.split(',"', maxsplit=1)[1]
        # les autres sont séparées par `","`
        cols = row.split('","')
        # Indices codés en dur, pas très général mais c'est pas grave
        mal.append(cols[9])
        ilo.append(cols[2])
    return mal, ilo

def write_list(lst, p):
    with open(p, "w") as out_stream:
        for elem in lst:
            out_stream.write(f"{elem}\n")

lines = read_that_ugly_csv("../../data/austronesian_swadesh.csv")
mal, ilo = get_malagasy_ilocano(lines)
write_list(mal, "mal.txt")
write_list(ilo, "ilo.txt")
```

```python
with open('swadesh_light.csv', 'w') as csvfile:
    fieldnames = ['english', 'ilocano']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|',quotechar='$')
    writer.writeheader()
    for num, en, ilo in swadesh_light:
        writer.writerow({'english': en, 'ilocano': ilo})

```

## ☕ Exos 12 ☕

> 1\. Écrire une fonction qui reçoit deux noms de langues austronésiennes, une liste de mots en
> anglais et renvoie chacun des mots anglais avec leur traduction dans les deux langues.

```python
import csv
import collections

def get_austro_words(lang1, lang2, words, path):
    """
    Reçoit un couple de langues (langue1, langue2) et une liste de mots (words)
    Cherche dans la liste Swadesh des langues austronésiennes les traductions des mots
    dans ces deux langues.
    Renvoie un dictionnaire {'langue1': [w1, w2], 'langue2': [w1, w2]}
    Liste vide si la langue n'est pas répertoriée dans la liste
    """
    res = collections.defaultdict(list)
    with open(path) as swadesh:
        reader = csv.DictReader(swadesh)
        if not (lang1 in reader.fieldnames):
            res[lang1] = []
        if not (lang2 in reader.fieldnames):
            res[lang2] = []
        for row in reader:
            if row["English"] in words:
                res[lang1].append(row[lang1])
                res[lang2].append(row[lang2])
        return res
```

> 2\. Pour chaque mot du Cebuano de la liste Swadesh austronésienne, trouvez les mots des autres
> langues qui ont les deux ou trois premiers caractères en commun. (optionnel si vous voulez jouer
> avec les expressions régulières) Si le mot commence par une voyelle, elle pourra différer dans les
> autres langues. Ex: isa / usa seront considérées comme similaires (i/u) parce qu'à part la
> première lettre voyelle elles sont similaires.

```python
def same_prefix(cebuano_word, word):
    """Vérifie si deux mots ont le même préfixe (longueur 2 ou 3)
    Si les premières lettres sont des voyelles on les considère similaires
    """
    if cebuano_word and word:
        if cebuano_word[0] in "aeiou" and word[0] in "eaiou":
            return cebuano_word[1:2] == word[1:2]
        else:
            return cebuano_word[0:2] == word[0:2]
    else:
        return False

def find_words_same_prefix(path):
    res = collections.defaultdict(list)
    with open(file) as swadesh:
        reader = csv.DictReader(swadesh)
        for row in reader:
            cebuano_w = row['Cebuano']
            for lang, cell in row.items():
                if lang == 'Cebuano':
                    continue
                for word in cell.split(','):# parce qu'on a des cellules avec plusieurs mots
                    if same_prefix(cebuano_w, word):
                        res[cebuano_w].append({lang:word})
    return res
```

> 3\. **Pour les champion⋅nes** Sans rechercher de solution sur internet, essayez d'implémenter une
   fonction qui calcule la distance de Levenshtein. (Vous pouvez chercher ce que c'est que la
   distance de Levenshtein et l'algorithme en pseudo-code, mais n'allez pas chercher directement
   d'implémentation en Python !).

Voir <http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/c_dist/edit_distance.html> et
<https://fr.wikipedia.org/wiki/Distance_de_Levenshtein> Pour les implémentations
: <https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python>


```python
def distance(longer_word, shorter_word):
    if len(longer_word) < len(shorter_word):
        shorter_word, longer_word = longer_word, shorter_word

    if longer_word == shorter_word:
        return 0
    elif len(longer_word) == 0:
        return len(shorter_word)
    elif len(shorter_word) == 0:
        return len(longer_word)
    else:
        matrix = {}
        longer_word = ' ' + longer_word
        shorter_word = ' ' + shorter_word
        W1 = len(longer_word)
        W2 = len(shorter_word)
        for i in range(W1):
            matrix[i, 0] = i
        for j in range (W2):
            matrix[0, j] = j
        for i in range(1, W1):
            for j in range(1, W2):
                if longer_word[i] == shorter_word[j]:
                    cost = 0
                else:
                    cost = 1
                matrix[i, j] = min(
                    matrix[i-1, j] + 1, # effacement
                    matrix[i, j-1] + 1, # insertion
                    matrix[i-1, j-1] + cost # substitution 
                    )
        return matrix[W1-1, W2-1]


def main():
    longer_word, shorter_word = ("roule", "raoul")
    print(f"{longer_word}, {shorter_word}")
    print(distance(longer_word, shorter_word))
```