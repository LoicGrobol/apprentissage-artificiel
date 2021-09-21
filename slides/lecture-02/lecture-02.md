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

Cours 2 : Structures de données
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22

## Correction des exos

Voir [https://github.com/LoicGrobol/web-interfaces/tree/main/corrections](https://github.com/LoicGrobol/web-interfaces/tree/main/corrections)  

## Les listes : fonctions

- Les listes héritent des fonctions des *sequences*, elles ont également des [fonctions
  propres](https://docs.python.org/3.6/tutorial/datastructures.html#more-on-lists)
- Parmi ces fonctions, nous utiliserons principalement :
  - `append(x)` : ajoute un élément x à la fin de la liste (haut de la pile*)
  - `extend([x, y, z])` : ajoute tous les éléments de la liste arg à la fin de la liste
  - `pop(index=-1)` : supprime et renvoie l'élément de la liste à la position `index`
  - `index(x)` : renvoie l'index du premier élément de valeur x
  - `count(x)` : renvoie le nombre de fois où x apparaît
  - `sort(key=None, reverse=False)` : trie et modifie la liste, lire la
    [doc](https://docs.python.org/3/howto/sorting.html#sortinghowto) pour en savoir plus sur les
    ordres de tri.

```python
stack = [12, 15, 12, 7, 18]
stack.index(12)
```

```python
stack.count(12)
```

**Attention** : si vous avez plusieurs éléments à compter, utilisez plutôt
[`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter)

```python
stack.sort()
stack
```

```python
stack.append(23)
stack
```

```python
stack.append([35, 46])
stack
```

```python
stack.extend([51, 52])
stack
```

### ✍️ Exo 1 ✍️

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

## Les listes en compréhension

- Elles permettent de définir des listes par filtrage ou opération sur les éléments d'une autre
  liste
- La [PEP 202](http://www.python.org/dev/peps/pep-0202/) conseille de préférer les listes en
  compréhension aux fonctions `map()` et `filter()`  
- C'est puissant et concis, *so pythonic*

```python
[i ** 2 for i in range(10)]
```

```python slideshow={"slide_type": "subslide"}
[i ** 2 for i in range(10) if i % 2 == 0]
```

```python
[(i, j) for i in range(2) for j in ['a', 'b']]
```

### ✍️ Exo 2 ✍️

Utilisez une liste en compréhension sur la sortie de votre fonction tokenize de manière à ne retenir
que les noms composés

```python
words = tokenize("De-ci de-là, cahin-caha, va trottine, va chemine, va petit âne")
compounds = [] # ← modifiez ça
assert compounds == [['De', 'ci'], ['de', 'là,'], ['cahin', 'caha,']]
```

## Parcours de liste

La boucle `for` est particulièrement adaptée pour parcourir les itérables et donc les listes

```python slideshow={"slide_type": "subslide"}
voyelles = ['a', 'e', 'i', 'o', 'u']
for item in voyelles:
    print(item)
```

La fonction `enumerate` peut être utile dans certains cas, elle renvoie un `tuple` contenant
l'indice et la valeur de l'item à l'indice concerné

```python
for i, item in enumerate(voyelles):
    print(i, item)
```

C'est de très loin préférable à itérer sur `range(len(voyelles))`.

## Copie

Dans `y = x`, `y` n'est pas une copie de x, les deux pointent vers le même objet. C'st
particulièrement important pour les objets *mutables* comme les listes.

```python
x = [1, 2, 3]
y = x
y[0] = 4
x
```

Si ce qu'on veut copier est une liste, on peut utiliser

```python
x = [1, 2, 3]
y = x[:]
```

ou

```python
y = list(x)
y[0] = 4
x
```

Il y a d'autres façons de faire. Pour les objets complexes on peut regarder du côté du module
[`copy`](https://docs.python.org/3/library/copy.html) mais il n'y a pas de réponse universelle et
copier c'est souvent coûteux. Le mieux à faire quand on a envie de faire une copie c'est de
commencer par se demander si on en a vraiment besoin.

# Déballage de séquences

- Le *sequence unpacking* permet d'effectuer plusieurs affectations simultanées
- L'*unpacking* s'applique souvent sur des tuples

```python
x, y, z = (1, 2, 3)
y
```

```python
lexique = [("maison", "mEz§"), ("serpent", "sERp@")]
for ortho, phon in lexique:
    print(phon)
```

- On peut aussi utiliser `*` pour déballer une séquence en argument de fonction

```python
bornes = (0, 10)
for i in range(*bornes):
    print(i)
```

## Les ensembles

Les ensembles (`set`) sont des collections non ordonnées d'élements sans doublons
Les ensembles supportent les fonctions mathématiques d'union, d'intersection, de différence ([doc](https://docs.python.org/3.6/library/stdtypes.html#set))

- `value in s` renvoie si `value` est un élément de `s`
- `union(*sets)` renvoie l'union de tous les `sets` (l'ensemble des valeurs contenues dans tous les
  sets).
- `intersection(*sets)` renvoie l'intersection de tous les `sets` (l'ensemble des valeurs contenues
  dans au moins un set).


```python
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

Attention il y a un piège

```python
a = {1}
type(a)
```

```python
b = {}
type(b)
```

## ✍️ Exo 3

Dans cet extrait de données tirées des [listes de Swadesh de langues
austronésiennes](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists), ici pour le
tagalog et le cebuano, trouvez les mots en commun.

```python
tagalog = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'tayo', 'you_pl':'kayo', 'they':'sila',\
           'this':'ito', 'that':'iyan', 'here':'dito', 'there':'doon', 'who':'sino',\
           'what':'ano', 'where':'saan', 'when':'kailan', 'how':'paano'}
cebuano = {'i':'ako', 'you_sg':'ikaw', 'he':'siya', 'we':'kita', 'you_pl':'kamo', 'they':'sila',\
           'this':'kiri', 'that':'kana', 'here':'diri', 'there':'diha', 'who':'kinsa',\
           'what':'unsa', 'where':'asa', 'when':'kanus-a', 'how':'unsaon'}
set(tagalog.values()).intersection(set(cebuano.values()))
```

## Les dictionnaires

- Les dictionnaires (`dict`) sont des structures de données associatives de type clé: valeur
- Les clés d'un dictionnaire sont uniques, seuls les types *hashable* (*immutable* et objets que
  vous avez définis) peuvent être des clés
  ([doc](https://docs.python.org/3.6/library/stdtypes.html#mapping-types-dict))

  - `key in d` renvoie True si `key` est une clé de `d`
  - `keys()` renvoie la liste des clés
  - `values()` renvoie la liste des valeurs
  - `items()` renvoie la liste des couples clé:valeur (tuple)
  - `get(key, default=None)` renvoie la valeur associée à `key`. Si `key` n'existe pas, renvoie
    l'argument `default`. Ne modifie pas le dictionnaire.
  - `setdefault(key, default=None)` si `key` n'existe pas, insère `key` avec la valeur `default`
    dans le dictionnaire puis renvoie la valeur associée à la clé.

```python
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

## Module collections

- Le module *collections* propose des implémentations de structures de données supplémentaires
- Dans la liste (voir [doc](https://docs.python.org/3.6/library/collections.html)), deux pourront
  nous intéresser :

  - `defaultdict`

     `defauldict` est similaire à un `dict` mais il permet l'autovivification

      Son implémentation le rend plus rapide qu'un dictionnaire utilisé avec la fonction
      `setdefault`


```python
import collections
lexique = [("couvent", "kuv"), ("couvent", "kuv@")]
dico = collections.defaultdict(list)
for ortho, phon in lexique:
    dico[ortho].append(phon)
dico
```

- `Counter`
  
`Counter` est un dictionnaire où les valeurs attendues sont les nombres d'occurences des clés

```python
from collections import Counter
cnt = Counter()
list = ['le', 'guépard', 'le', 'poursuit']
for item in list:
    cnt[item] += 1
cnt
```

### ✍️ Exo 4

Faites la même chose avec un dictionnaire


## Les fichiers

- Pour travailler avec les fichiers on doit procéder à trois opérations :
   1. Ouverture avec la fonction `open` (lève l'exception `FileNotFoundError` en cas d'échec)
   2. Lecture (`read` ou `readline` ou `readlines`) et/ou écriture (`write`)
   3. Fermeture du fichier avec la fonction `close`
- Ouverture
  - `open` est une fonction qui accepte de nombreux arguments : RTFM
  - `open` renvoie un objet de type `file`
  - Le plus souvent elle s'emploie de la manière suivante:
  ```python
    >>> #f = open(filename, mode)	   
    >>> f = open('nom_fichier', 'w')
  ```

Les modes sont : 

- `r` : lecture (défaut)
- `w` : écriture
- `x` : création et écriture (échec si le fichier existe déjà)
- `a` : concaténation (append)
  
  
- `b` : mode binaire
- `t` : mode texte (défaut)
- `+` : read/write (ex: r+b)

## Les fichiers : ouverture

La documentation de Python conseille cette façon de faire :
```python
with open('mon_fichier', 'r') as f:
    read_data = f.read()
```
L'utilisation du mot clé `with` garantit la fermeture du fichier même si une exception est soulevée.  

## Les fichiers : lecture

- `read(size=-1)` lit les `size` premiers octets (mode `b`) ou caractères (mode `t`). Si `size` < 0,
  lit tout le fichier.
- `readline(size=-1)` lit au plus `size` caractères ou jusqu'à la fin de ligne. Si `size` < 0, lit
  toute la ligne. Il est conseillé de ne pas toucher à `size`.
- `readlines(hint=-1)` lit `hint` lignes du fichier. Si `hint` < 0, lit toutes les lignes du
  fichier.
- un objet `file` est un itérable ! (*the pythonic way*)

```python
for line in f:
    process(line)
```

## Les fichiers : écriture et fermeture

- `write(text)` écrit `texte` dans le fichier?
- `close()` ferme le fichier.  

En règle générale veillez à toujours fermer les objets fichiers.  
En mode écriture oublier de fermer un fichier peut réserver des mauvaises surprises

- fonction `print`
```python
with open('mon_fichier', 'w') as output_f:
    for item in words:
        print(item, file=output_f)
```
- `sys.stdin`, `sys.stdout` et `sys.stderr` sont des objets de type `file`

### ✍️ Exo 5

Lisez le fichier `data/austronesian_swadesh.csv` et écrivez les mots des langues Ilocano et Malagasy dans deux fichiers distincts.  
Les données viennent de [Wiktionary](https://en.wiktionary.org/wiki/Appendix:Austronesian_Swadesh_lists).

(Essayez de faire comme si vous ne connaissiez pas le module csv sinon la partie qui suit n'aura aucun intérêt.)

```python
# c'est compliqué sans le module csv quand même
```

## Module csv

La documentation est ici : [https://docs.python.org/3/library/csv.html](https://docs.python.org/3/library/csv.html)  
Parce que les données au format csv sont très répandues et parce qu'il peut être pénible de le lire correctement, le module csv est là pour vous aider.  
Pour le dire vite il y a deux façons de l'utiliser : reader/writer ou DictReader/DictWriter.


- `csv.reader`

```python
import csv

swadesh_light = []
with open('data/austronesian_swadesh.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"') # à l'ouverture je spécifie les séparateur de champ et de chaîne  
    for row in reader: # l'objet reader est un itérable
        swadesh_light.append(row[0:3])
        print(' | '.join(row[0:3])) # row est une liste de chaînes de caractères
```

- `csv.writer`

```python
with open('swadesh_light.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='|',quotechar='"')
    #writer.writerows(swadesh_light) ici on écrit tout en une fois
    for item in swadesh_light:
        writer.writerow(item) # writerow reçoit une liste de chaînes
```

- csv.DictReader  
Cette classe s'appuie sur la ligne d'en-tête pour créer une suite de dictionnaires.  
S'il n'y a pas de ligne d'en-tête on peut utiliser une liste `fieldnames` en paramètre.

```python
with open('data/austronesian_swadesh.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',',quotechar='"')
    for row in reader: # ici row est un dictionnaire
         print(f"{row['Ilocano']} | {row['Malagasy']}")

```

- csv.DictWriter  
  Cette fois il s'agit de générer un fichier csv à partir d'une séquence de dictionnaires. Le paramètre `fieldnames` est obligatoire.

```python
with open('swadesh_light.csv', 'w') as csvfile:
    fieldnames = ['english', 'ilocano']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|',quotechar='$')
    writer.writeheader()
    for num, en, ilo in swadesh_light:
        writer.writerow({'english': en, 'ilocano': ilo})

```

## Module `re`

```python
import re
```

<!-- #region -->
- `re` est un module particulièrement important, vous devez lire la [doc](https://docs.python.org/3/library/re.html), absolument

- La doc officielle est parfois aride, ce [howto](https://docs.python.org/3.6/howto/regex.html) rédigé par A.M. Kuchling est plus digeste


a minima vous devez connaître les fonctions :

- `findall` : trouve toutes les occurences du motif, retourne une liste de chaînes trouvées
- `search` : trouve le motif, retourne un objet `Match`, `None` sinon
- `match` : détermine si le motif est présent au début de la chaîne, retourne un objet `Match`,
  `None` sinon
- `split` : découpe une chaîne selon un motif, retourne une liste de chaînes
- `sub` : remplace les occurences d'un motif par une chaîne de remplacement
- `compile` : compilation d'un motif (pattern), retourne un objet `Pattern`
<!-- #endregion -->

```python
if re.search(r"(\w|\s)+", "Un léopard me pourchasse"):
    print("Cours !")
```

```python
re.sub(r'e|é', 'i', 'éléphanteau')
```

## `\w` et Python3

`\w` est la classe prédéfinie des caractères alphanumériques :

- En Python 2 `\w` ~correspond~ correspondait à `[A-Za-z0-9_]`, avec les locales il est possible d'y
  ajouter d'autres caractères
- En Python 3 `\w` correspond à tous les caractères qui ont la propriété Unicode Letter d'après le
  module `unicodedata` (sauf si le motif est compilé en binaire ou si l'option `re.ASCII` est
  activée)

```python
if re.search(r"\w", "馬青區團長成中央代表"):
    print("Yeah !")
```

```python
if re.search(r"\w", "هيلاري كلينتون"):
    print("Yeah !")
```

### ☕ Exos 6 ☕


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

3\. Sans rechercher de solution sur internet, essayez d'implémenter une fonction qui calcule la
   distance de Levenshtein