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

<!-- LTeX: language=fr -->

Cours 1 : corrections
=====================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-09-28

## ✂️ Exo 1 ✂️

1\. Écrire une fonction `crude_tokenizer` qui prend comme argument une chaine de caractères et
    renvoie la liste des mots de cette chaîne en séparant sur les espaces.

```python
def crude_tokenizer(s):
    return s.split()

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', "l'existence", 'du', 'kiwi-fruit.'
]
```

2\. Modifier la fonction `crude_tokenizer` pour qu'elle sépare aussi suivant les caractères
   non alphanumériques. **Indice** ça peut être utile de revoir [la doc sur les expressions
   régulières](https://docs.python.org/3/library/re.html) ou de relire [un tuto à ce
   sujet](https://realpython.com/regex-python/).

```python
import re
def crude_tokenizer(s):
    return [w for w in re.split(r"\s|\W", s.strip()) if w]

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', 'l', 'existence', 'du', 'kiwi', 'fruit'
]
```

3\. On aimerait maintenant garder les apostrophes à la fin du mot qui les précède, ainsi que les
   mots composés ensemble.

```python
import re  # Si jamais on a pas exécuté la cellule précédente
def crude_tokenizer(s):
    return re.findall(r"\b\w+?\b(?:'|(?:-\w+?\b)*)?", s)

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit'
]
```

## 🔢 Exo 2 🔢

Écrire une fonction `extraire_bigrammes` qui prend en entrée une chaine de caractères, la tokenize
avec `crude_tokenizer` et renvoie la liste des bigrammes correspondants sous forme de couples de
mots.


Version directe

```python
def extraire_bigrammes(s):
    tokenized = crude_tokenizer(s)
    res = []
    for i in range(len(tokenized)-1):
        res.append((tokenized[i], tokenized[i+1]))
    return res

assert extraire_bigrammes("Je reconnais l'existence du kiwi-fruit.") == [
    ('Je', 'reconnais'),
     ('reconnais', "l'"),
     ("l'", 'existence'),
     ('existence', 'du'),
     ('du', 'kiwi-fruit')
]
```

Version artistique

```python
def extraire_bigrammes(s):
    tokenized = crude_tokenizer(s)
    return list(zip(tokenized[:-1], tokenized[1:]))

assert extraire_bigrammes("Je reconnais l'existence du kiwi-fruit.") == [
    ('Je', 'reconnais'),
     ('reconnais', "l'"),
     ("l'", 'existence'),
     ('existence', 'du'),
     ('du', 'kiwi-fruit')
]
```

Si vous trouvez ça obscur essayez le code ci-dessous, et allez voir ce qu'il donne [sur Python
Tutor](https://pythontutor.com/render.html#code=tokenized%20%3D%20%5B'Je',%20'reconnais',%20%22l'%22,%20'existence',%20'du',%20'kiwi-fruit'%5D%0A%0Afirst_words%20%3D%20tokenized%5B%3A-1%5D%0Asecond_words%20%3D%20tokenized%5B1%3A%5D%0A%0Afor%20t%20in%20zip%28first_words,%20second_words%29%3A%0A%20%20%20%20print%28t%29&cumulative=false&curInstr=10&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false)

```python
tokenized = ['Je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']

first_words = tokenized[:-1]
second_words = tokenized[1:]

print(first_words)
print(second_words)

for t in zip(first_words, second_words):
    print(t)
```

### Calculer les probas


On va ensuite estimer les probas de générer un certain mot $w_1$ sachant que le mot précédent est
$w_0$. On le fait en utilisant la formule du maximum de vraisemblance:

\begin{equation}
   P(w_1|w_0) = \frac{\text{nombre d'occurrences du bigramme $w_0 w_1$}}{\text{nombre d'occurrences de l'unigramme $w_0$}}
\end{equation}

Pour que ce soit plus agréable à sampler on va utiliser un dictionnaire de dictionnaires :
`probs[v][w]` stockera $P(w|v)$.

```python
from collections import defaultdict

probs = defaultdict(dict)
for (v, w), c in bigrams.items():
    probs[v][w] = c/unigrams[v]

# Pour ne pas masquer des erreurs pendant le sampling, on en refait un dict normal
probs = dict(probs)
probs
```

Un autre truc un peu pénible, c'est qu'en tenant compte de la casse comme on le fait, on sépare en
deux les comptes de chaque mot (suivant qu'il se trouve ou non en début de phrase). C'est pas
complètement une erreur, mais c'est un peu désagréable, on va normaliser tout ça.

```python
def crude_tokenizer_and_normalizer(s):
    return [w.lower() for w in re.split(r"\s|(\W)", s.strip()) if w]
```

### Générer

Pour l'instant on ne va pas se préoccuper de sauvegarder le modèle on va l'utiliser directement pour
sampler. Le principe est simple : on sample le premier mot, puis on sample le deuxième mot en
prenant le premier qu'on vient de générer et ainsi de suite.


Est-ce que vous voyez le problème ?


Comment on sample le premier mot ?

Et quand est-ce qu'on décide de s'arrêter ?


On rouvre le bouquin et on trouve

>  We’ll first need to augment each sentence with a special symbol `<s>` at the beginning of the
> sentence, to give us the bigram context of the first word. We’ll also need a special end-symbol.
> `</s>`


Oups


Allez, on corrige

```python
l = [1,2,3,4,5]
l2 = [-1, 0, *l]
l2
```

```python
unigrams = Counter()
bigrams = Counter()
with open("../../data/zola_ventre-de-paris.txt") as in_stream:
    for line in in_stream:
        words = crude_tokenizer_and_normalizer(line.strip())
        if "<s>" in words or "</s>" in words:
            raise ValueError(f"Symboles de début/fin de phrases déjà présents dans le corpus {line!r}")
        unigrams.update(("<s>", *words, "</s>"))
        bigrams.update(zip(("<s>", *words), (*words, "</s>")))

probs = defaultdict(dict)
for (v, w), c in bigrams.items():
    probs[v][w] = c/unigrams[v]

probs = dict(probs)
probs
```

Il y a encore un petit problème

```python
probs["<s>"]["</s>"]
```

🤔


On a compté les lignes vides 😤. Ça ne posait pas de problème jusque-là puisque ça n'ajoutait rien
aux compteurs de n-grammes, mais maintenant ça nous fait des `["<s>", "</s>"]`.


C'est reparti

```python
unigrams = Counter()
bigrams = Counter()
with open("../../data/zola_ventre-de-paris.txt") as in_stream:
    for line in in_stream:
        # Voi-là
        if line.isspace():
            continue
        words = crude_tokenizer_and_normalizer(line.strip())
        # Pourquoi on fait ça ?
        if "<s>" in words or "</s>" in words:
            raise ValueError(f"Symboles de début/fin de phrases déjà présents dans le corpus {line!r}")
        unigrams.update(("<s>", *words, "</s>"))
        bigrams.update(zip(("<s>", *words), (*words, "</s>")))

probs = defaultdict(dict)
for (v, w), c in bigrams.items():
    probs[v][w] = c/unigrams[v]

probs = dict(probs)
probs
```

### Générer pour de vrai

**Bon c'est bon maintenant ?**


À peu près. On va pouvoir sampler.


Pour ça on va piocher dans le module [`random`](https://docs.python.org/3/library/random.html) de la
bibliothèque standard, et en particulier la fonction
[`random.choices`](https://docs.python.org/3/library/random.html#random.choices) qui permet de tirer
au sort dans une population finie en précisant les probabilités de chacun de éléments. Le poids
n'ont en principe pas besoin d'être normalisés (mais ils le seront ici, évidemment).

```python
import random
```

Voyons déjà comment choisir le premier mot

```python
candidates = list(probs["<s>"].keys())
#  On pourrait faire plus fancy avec `zip`, cherchez comment
weights = [probs["<s>"][c] for c in candidates] 
random.choices(candidates, weights)[0]  # `choices` renvoit une liste, voir sa doc
```

Ça marche, maintenant une phrase ! On sample mot par mot et on s'arrête quand on arrive à `</s>`

```python
sent = ["<s>"]
while sent[-1] != "</s>":
    candidates = list(probs[sent[-1]].keys())
    weights = [probs[sent[-1]][c] for c in candidates]
    sent.append(random.choices(candidates, weights)[0])

print(" ".join(sent[1:-1]))
```

C'est rigolo, hein ?


Qu'est-ce que vous pensez des textes qu'on génère ?

### Les trigrammes

Avant de généraliser, on va voir comment passer aux trigrammes

```python
bigrams = Counter()
trigrams = Counter()
with open("../../data/zola_ventre-de-paris.txt") as in_stream:
    for line in in_stream:
        if line.isspace():
            continue
        words = crude_tokenizer_and_normalizer(line.strip())
        if "<s>" in words or "</s>" in words:
            raise ValueError(f"Symboles de début/fin de phrases déjà présents dans le corpus {line!r}")
        words = ["<s>", "<s>", *words, "</s>"]
        bigrams.update(zip(words[:-1], words[1:]))
        # On pourrait faire comme avec les bigrammes mais ça généralisera mieux comme ça
        # À votre avis pourquoi des tuples ?
        trigrams.update((tuple(words[i-2:i]), w) for i, w in enumerate(words[2:], start=2))

probs = defaultdict(dict)
for ((u, v), w), c in trigrams.items():
    probs[(u, v)][w] = c/bigrams[(u, v)]

probs = dict(probs)
probs
```

```python
sent = ["<s>", "<s>"]
while sent[-1] != "</s>":
    candidates = list(probs[tuple(sent[-2:])].keys())
    weights = [probs[tuple(sent[-2:])][c] for c in candidates]
    sent.append(random.choices(candidates, weights)[0])

print(" ".join(sent[2:-1]))
```

## Les n-grammes

On passe aux n-grammes ? On va essayer de les faire de façon un peu plus compacte.

```python
def get_ngrams_probs(path, n=2):
    ngrams = defaultdict(Counter)
    with open(path) as in_stream:
        for line in in_stream:
            if line.isspace():
                continue
            words = crude_tokenizer_and_normalizer(line.strip())
            if "<s>" in words or "</s>" in words:
                raise ValueError(f"Symboles de début/fin de phrases déjà présents dans le corpus {line!r}")
            words = [*("<s>" for _ in range(n-1)), *words, "</s>"]
            for i, w in enumerate(words[n-1:], start=n-1):
                ngrams[tuple(words[i-n+1:i])][w] += 1
    probs = defaultdict(dict)
    for trigger, targets in ngrams.items():
        trigger_occurences = sum(targets.values())
        for t, c in targets.items():
            probs[trigger][t] = c/trigger_occurences
    return dict(probs)
```

```python
get_ngrams_probs("../../data/zola_ventre-de-paris.txt", 4)
```

```python
def sample_from_probs(probs, n):
    # On pourrait inférer n automatiquement mais fleeeeemme
    sent = ["<s>" for _ in range(n-1)]
    while sent[-1] != "</s>":
        candidates = list(probs[tuple(sent[-n+1:])].keys())
        weights = [probs[tuple(sent[-n+1:])][c] for c in candidates]
        sent.append(random.choices(candidates, weights)[0])
    return " ".join(sent[n-1:-1])
```

```python
probs = get_ngrams_probs("../../data/zola_ventre-de-paris.txt", 4)
```

```python
sample_from_probs(probs, 4)
```

Vous voyez un problème ?

## Un peu d'originalité

Le modèle ici marche, mais comme le corpus est un peu petit, il manque souvent d'originalité pour
des grandes valeurs de $n$. Il y a plusieurs façons d'y remédier et les sections 3.4 et 3.5 de
*Speech and Language Processing* donnent plus de détails à ce sujet.
