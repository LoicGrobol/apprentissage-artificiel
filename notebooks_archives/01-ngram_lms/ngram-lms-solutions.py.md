---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

Modèles de langues à n-grammes : corrections
============================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-09-28

## ✂️ Tokenization ✂️

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

4\. Écrire une fonction `crude_tokenizer_and_normalizer` qui en plus de tokenizer comme précédemment
met tous les mots en minuscules

On peut évidemment copier-coller le code au-dessus, mais on peut aussi réutiliser ce qu'on a déjà
défini :

```python
def crude_tokenizer_and_normalizer(s):
    return crude_tokenizer(s.lower())

assert crude_tokenizer_and_normalizer("Je reconnais l'existence du kiwi-fruit.") == [
    'je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit'
]
```

## 💜 Extraire les bigrammes 💜

Écrire une fonction `extract_bigrams` qui prend en entrée une liste de mots et renvoie la liste des
bigrammes correspondants sous forme de couples de mots.


Version directe

```python
def extract_bigrams(words):
    res = []
    for i in range(len(words)-1):
        res.append((words[i], words[i+1]))
    return res

assert extract_bigrams(['je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']) == [
    ('je', 'reconnais'),
     ('reconnais', "l'"),
     ("l'", 'existence'),
     ('existence', 'du'),
     ('du', 'kiwi-fruit')
]
```

Version artistique

```python
def extract_bigrams(words):
    return list(zip(words[:-1], words[1:]))

assert extract_bigrams(['je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']) == [
    ('je', 'reconnais'),
     ('reconnais', "l'"),
     ("l'", 'existence'),
     ('existence', 'du'),
     ('du', 'kiwi-fruit')
]
```

Si vous trouvez ça obscur essayez le code ci-dessous, et allez voir ce qu'il donne [sur Python Tutor](https://pythontutor.com/render.html#code=tokenized%20%3D%20%5B'Je',%20'reconnais',%20%22l'%22,%20'existence',%20'du',%20'kiwi-fruit'%5D%0A%0Afirst_words%20%3D%20tokenized%5B%3A-1%5D%0Asecond_words%20%3D%20tokenized%5B1%3A%5D%0A%0Afor%20t%20in%20zip%28first_words,%20second_words%29%3A%0A%20%20%20%20print%28t%29&cumulative=false&curInstr=10&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false)

```python
tokenized = ['je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']

first_words = tokenized[:-1]
second_words = tokenized[1:]

print(first_words)
print(second_words)

for t in zip(first_words, second_words):
    print(t)
```


## 🔢 Compter 🔢

Écrire une fonction `read_corpus` qui prend en argument un chemin vers un fichier texte, l'ouvre, le
tokenize et y compte les unigrammes et les bigrammes en renvoyant deux `Counter` associant
respectivement à chaque mot et à chaque bigramme leurs nombres d'occurrences.

```python
from collections import Counter
    
def read_corpus(file_path):
    unigrams = Counter()
    bigrams = Counter()
    with open(file_path) as in_stream:
        for line in in_stream:
            words = crude_tokenizer_and_normalizer(line.strip())
            unigrams.update(words)
            bigrams.update(extract_bigrams(words))
    
    return unigrams, bigrams


unigram_counts, bigram_counts = read_corpus("data/zola_ventre-de-paris.txt")

assert unigram_counts.most_common(4) == [('de', 5292), ('la', 3565), ('les', 2746), ('il', 2443)]
assert bigram_counts.most_common(4) == [
    (('de', 'la'), 754),
     (("qu'", 'il'), 424),
     (('à', 'la'), 336),
     (("d'", 'une'), 321)
]
```

## 🤓 Estimer les probas 🤓

Écrire une fonction `get_probs`, qui prend en entrée les compteurs de bigrammes et
d'unigrammes et renvoie le dictionnaire `probs`

```python
def get_probs(unigram_counts, bigàram_counts):
    probs = dict()
    for (v, w), c in bigram_counts.items():
        if v not in probs:
            # Si on a pas encore rencontré de bigrammes commençant par `v`, il faut
            # commencer par créer `probs[v]`
            probs[v] = dict()
        probs[v][w] = c/unigram_counts[v]
    return probs

probs = get_probs(unigram_counts, bigram_counts)
assert probs["je"]["déjeune"] == 0.002232142857142857
```

Avec `collections.defaultdict` :

```python
from collections import defaultdict

def get_probs(unigram_counts, bigram_counts):
    # Un dictionnaire de dictionnaires créés automatiquement à l'accès,
    # Ça évite de faire un test et ça rend souvent le code plus lisible
    probs = defaultdict(dict)
    for (v, w), c in bigram_counts.items():
        probs[v][w] = c/unigram_counts[v]

    # Pour ne pas masquer des erreurs pendant le sampling, on en refait un dict normal
    return dict(probs)

probs = get_probs(unigram_counts, bigram_counts)
assert probs["je"]["déjeune"] == 0.002232142857142857
```

## 💁🏻 Générer un mot 💁🏻

> À vous de jouer : écrire une fonction `gen_next_word` qui prend en entrée le dictionnaire `probs`
> et un mot et renvoie en sortie un mot suivant, choisi en suivant les probabilités estimées
> précédemment

```python
import random

def gen_next_word(probs, prompt):
    # On convertit en liste pour s'assurer que les mots et les poids sont bien dans le même ordre
    candidates = list(probs[prompt].keys())
    weights = [probs[prompt][c] for c in candidates]
    return random.choices(candidates, weights)[0]
```


## 🤔 Générer 🤔

1\. Modifier `read_corpus` pour ajouter à la volée `<s>` au début de chaque ligne et `</s>`
à la fin de chaque ligne.

```python
def read_corpus(file_path):
    unigrams = Counter()
    bigrams = Counter()
    with open(file_path) as in_stream:
        for line in in_stream:
            words = crude_tokenizer_and_normalizer(line.strip())
            words.insert(0, "<s>")
            words.append("</s>")
            unigrams.update(words)
            bigrams.update(extract_bigrams(words))
    
    return unigrams, bigrams


unigram_counts, bigram_counts = read_corpus("data/zola_ventre-de-paris.txt")

assert unigram_counts.most_common(4) == [('<s>', 8945), ('</s>', 8945), ('de', 5292), ('la', 3565)]
assert bigram_counts.most_common(4) == [
    (('<s>', '</s>'), 1811),
    (('<s>', 'il'), 775),
    (('de', 'la'), 754),
    (('<s>', 'elle'), 576)
]
```

2\. Modifier `read_corpus` pour ignorer les lignes vides

```python
def read_corpus(file_path):
    unigrams = Counter()
    bigrams = Counter()
    with open(file_path) as in_stream:
        for line in in_stream:
            if line.isspace():
                continue
            words = crude_tokenizer_and_normalizer(line.strip())
            words.insert(0, "<s>")
            words.append("</s>")
            unigrams.update(words)
            bigrams.update(extract_bigrams(words))
    
    return unigrams, bigrams


unigram_counts, bigram_counts = read_corpus("data/zola_ventre-de-paris.txt")

assert unigram_counts.most_common(4) == [('<s>', 7145), ('</s>', 7145), ('de', 5292), ('la', 3565)]
assert bigram_counts.most_common(4) == [
    (('<s>', 'il'), 775),
    (('de', 'la'), 754),
    (('<s>', 'elle'), 576),
    (("qu'", 'il'), 424)
]

probs = get_probs(unigram_counts, bigram_counts)
assert probs["<s>"]["le"] == 0.0298110566829951
```

## 😌 Générer pour de vrai 😌

Écrire une fonction `sample` qui prend en argument les probabilités de bigrammes (sous la forme d'un
dictionnaire de dictionnaires comme notre `prob`) et génère une phrase en partant de `<s>` et en
ajoutant des mots itérativement, s'arrêtant quand `</s>` a été choisi.

```python
import random

def generate(bigram_probs):
    sent = ["<s>"]
    while sent[-1] != "</s>":
        sent.append(gen_next_word(bigram_probs, sent[-1]))
    return sent
```

Pas de `assert` ici comme on a de l'aléatoire, mais la cellule suivante permet de tester si ça
marche :

```python
print(generate(probs))
```

Et ici pour avoir du texte qui ressemble à quelque chose :

```python
print(" ".join(generate(probs)[1:-1]))
```

## 🧐 Aller plus loin 🧐


### 3️⃣ Trigrammes 3️⃣

Coder un générateur de phrases à partir de trigrammes.

On va reprendre les mêmes idées qu'avant, cette fois sans tomber dans les pièges, ça devrait aller
plus vite ! La seule différence, c'est qu'au lieu de compter les bigrammes on compte les trigrammes
afin de pouvoir choisir chaque mot en fonction des deux précédents.

Un petit changement supplémentaire pour choisir le premier mot : au lieu d'un seul marqueur de début
de phrase `<s>`, on va maintenant devoir en mettre deux `<s> <s>`. Par contre, il suffit toujours
d'un seul `</s>`, vous suivez ?

```python
def extract_trigrams(words):
    res = []
    for i in range(len(words)-2):
        res.append((words[i], words[i+1], words[i+2]))
    return res

assert extract_trigrams(['je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']) == [
    ('je', 'reconnais', "l'",),
    ('reconnais', "l'", "existence"),
    ("l'", 'existence', "du"),
    ('existence', 'du', "kiwi-fruit"),
]
```

```python
def read_corpus_for_trigrams(file_path):
    unigrams = Counter()
    trigrams = Counter()
    with open(file_path) as in_stream:
        for line in in_stream:
            words = crude_tokenizer_and_normalizer(line.strip())
            words = ["<s>", "<s>"] + words
            words.append("</s>")
            unigrams.update(words)
            trigrams.update(extract_trigrams(words))
    
    return unigrams, trigrams
```


```python
def get_trigram_probs(unigram_counts, trigram_counts):
    probs = defaultdict(dict)
    for (w_1, w_2, w_3), c in trigram_counts.items():
        probs[(w_1, w_2)][w_3] = c/unigram_counts[w_3]

    return dict(probs)
```

```python
def generate_from_trigrams(trigram_probs):
    sent = ["<s>", "<s>"]
    while sent[-1] != "</s>":
        candidates = list(trigram_probs[(sent[-2], sent[-1])].keys())
        weights = [trigram_probs[(sent[-2], sent[-1])][c] for c in candidates]
        sent.append(random.choices(candidates, weights)[0])
    return sent
```

Et pour tester

```python
unigram_counts, trigram_counts = read_corpus_for_trigrams("data/zola_ventre-de-paris.txt")
trigram_probs = get_trigram_probs(unigram_counts, trigram_counts)
```

```python
print(" ".join(generate_from_trigrams(trigram_probs)[2:-1]))
```

### 🇳 N-grammes 🇳

Toujours la même chose, simplement il va falloir réfléchir un peu pour généraliser :

```python
def extract_ngrams(words, n):
    res = []
    for i in range(len(words)-n+1):
        # Tuple pour pouvoir s'en servir comme clé de dictionnaire et donc OK avec `Counter`
        res.append(tuple(words[i:i+n]))
    return res
```

```python
def read_corpus_for_ngrams(file_path, n):
    unigrams = Counter()
    ngrams = Counter()
    with open(file_path) as in_stream:
        for line in in_stream:
            words = crude_tokenizer_and_normalizer(line.strip())
            # Il nous faut bien `n-1` symboles de début de phrase 
            words = ["<s>"]*(n-1) + words
            words.append("</s>")
            unigrams.update(words)
            ngrams.update(extract_ngrams(words, n))
    
    return unigrams, ngrams
```


```python
def get_ngram_probs(unigram_counts, ngram_counts):
    probs = defaultdict(dict)
    for ngram, c in ngram_counts.items():
        probs[tuple(ngram[:-1])][ngram[-1]] = c/unigram_counts[ngram[-1]]

    return dict(probs)
```

On peut aussi écrire ça comme ceci avec un
[*unpacking*](https://stackabuse.com/unpacking-in-python-beyond-parallel-assignment/) (voir aussi
[la doc](https://docs.python.org/3/reference/expressions.html#expression-lists) pour la syntaxe
abstraite et la [PEP 448](https://peps.python.org/pep-0448/) qui l'a introduite)

```python
def get_ngram_probs(unigram_counts, ngram_counts):
    probs = defaultdict(dict)
    for ngram, c in ngram_counts.items():
        *previous_words, target_word = ngram
        probs[tuple(previous_words)][target_word] = c/unigram_counts[target_word]

    return dict(probs)
```

voire même

```python
def get_ngram_probs(unigram_counts, ngram_counts):
    probs = defaultdict(dict)
    for (*previous_words, target_word), c in ngram_counts.items():
        probs[tuple(previous_words)][target_word] = c/unigram_counts[target_word]

    return dict(probs)
```

```python
def generate_from_ngrams(ngram_probs, n):
    # On pourrait deviner `n` à partir de `ngram_probs…
    sent = ["<s>"] * (n-1)
    while sent[-1] != "</s>":
        # Essayer de bien réfléchir pour comprendre le `1-n`
        previous_words = tuple(sent[1-n:])
        candidates = list(ngram_probs[previous_words].keys())
        weights = [ngram_probs[previous_words][c] for c in candidates]
        sent.append(random.choices(candidates, weights)[0])
    return sent
```

Et pour tester

```python
n = 5
unigram_counts, ngram_counts = read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", n)
ngram_probs = get_ngram_probs(unigram_counts, ngram_counts)
```

```python
print(" ".join(generate_from_ngrams(ngram_probs, n)[n-1:-1]))
```
