---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[comment]: <> "LTeX: language=fr"

<!-- #region slideshow={"slide_type": "slide"} -->
Cours 8 : Modèles de langues à n-grammes
========================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-11
<!-- #endregion -->

```python
from IPython.display import display
```

## Pitch

On va apprendre un modèle de langues à n-grammes. On se basera pour la théorie et les notations sur
le chapitre 3 de [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) de
Daniel Jurafsky et James H. Martin (garde le donc pas loin). Notre objectif ici sera de faire du
*sampling*.

Pour les données on va d'abord travailler avec [Le Ventre de
Paris](../../data/zola_ventre-de-paris.txt) qui est déjà dans ce repo pour les tests puis avec [le
corpus CIDRE](https://www.ortolang.fr/market/corpora/cidre) pour passer à l'échelle, mais on
pourrait aussi utiliser Wikipedia (par exemple en utilisant
[WikiExtractor](https://github.com/attardi/wikiextractor)) ou [OSCAR](https://oscar-corpus.com/).

On va devoir faire les choses suivantes

- Extraire les unigrammes et le n-grammes d'un corpus (pour un certain n)
- Calculer les probas normalisées des bigrammes
- Les sauvegarder (par exemple dans un TSV)
- Sampler des phrases à partir du modèle
- (En option) évaluer le modèle sur un corpus de test
- Wrapper tout ça dans des jolis scripts

On va essayer de faire les choses à la main, sans trop utiliser de bibliothèques, pour bien
comprendre ce qui se passe.

## Premier prototype.


On va commencer par faire en entier le cas des bigrammes sur *Le Ventre de Paris* et on généralisera
ensuite.

### Lire et compter


On commence par lire un fichier et en extraire les unigrammes (ce qui nous donne le vocabulaire) et
les bigrammes. On va pour l'instant faire ça très basiquement avec une bête tokenisation sur les
espaces et les signes de ponctuation.

```python
import re
def poor_mans_tokenizer(s):
    return [w for w in re.split(r"\s|(\W)", s.strip()) if w]
```

Vous voyez pour quoi on ne fait pas simplement un `split()` ?

```python
from collections import Counter
unigrams = Counter()
bigrams = Counter()
with open("../../data/zola_ventre-de-paris.txt") as in_stream:
    for line in in_stream:
        words = poor_mans_tokenizer(line.strip())
        unigrams.update(words)
        bigrams.update(zip(words[:-1], words[1:]))
display(unigrams.most_common(10))
display(bigrams.most_common(10))
```

(Si vous trouvez `zip(words[:-1], words[1:])` obscur, faites quelques tests pour voir pourquoi ça marche.)

### Calculer les probas


On va ensuite estimer les probas de générer un certain mot $w_1$ sachant que le mot précédent est
$w_0$. On le fait en utilisant la formule du maximum de vraissemblance:

\begin{equation}
   P(w_1|w_0) = \frac{\text{nombre d'occurences du bigramme $w_0 w_1$}}{\text{nombre d'occurrences de l'unigramme $w_0$}}
\end{equation}

Pour que ce soit plus agréable à sampler on va utiliser un dictionnaire de dictionnaires :
`prob[v][w]` stockera $P(w|v)$.

```python
from collections import defaultdict

probs = defaultdict(dict)
for (v, w), c in bigrams.items():
    probs[v][w] = c/unigrams[v]

# Pour ne pas masquer des erreurs pendant le sampling, on en refait un dict normal
probs = dict(probs)
probs
```

Un autre truc un peu pénible c'est qu'en tenant compte de la casse comme on le fait, on sépare en
deux les comptes de chaque mot (suivant qu'il se trouve ou non en début de phrase). C'est pas
complètement une erreur mais c'est un peu désagréable, on va normaliser tout ça.

```python
def poor_mans_tokenizer_and_normalizer(s):
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
unigrams = Counter()
bigrams = Counter()
with open("../../data/zola_ventre-de-paris.txt") as in_stream:
    for line in in_stream:
        words = poor_mans_tokenizer_and_normalizer(line.strip())
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
        words = poor_mans_tokenizer_and_normalizer(line.strip())
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
        words = poor_mans_tokenizer_and_normalizer(line.strip())
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
            words = poor_mans_tokenizer_and_normalizer(line.strip())
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
*Speech and Language Processing donnent plus de détails à ce sujet.
