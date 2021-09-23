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

Cours 2 : corrections
=====================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-09-22

```python slideshow={"slide_type": "slide"}
from IPython.display import display
```

## ✍️ Exo 1 ✍️

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

### ✍️ Exo 2 ✍️

> Utilisez une liste en compréhension sur la sortie de votre fonction tokenize de manière à ne
> retenir que les noms composés

```python
words = tokenize("De-ci de-là, cahin-caha, va trottine, va chemine, va petit âne")
compounds = [w for w in words if not isinstance(word, str)]
assert compounds == [['De', 'ci'], ['de', 'là,'], ['cahin', 'caha,']]
```

Là encore on pourrait mieux faire, mais ça ne vaut pas vraiment le coup pour cet exo.

## ✍️ Exo 3

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
### ✍️ Exo 4


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

### ✍️ Exo 5

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

## ☕ Exos 6 ☕

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

> 3\. Sans rechercher de solution sur internet, essayez d'implémenter une fonction qui calcule la
> distance de Levenshtein

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
