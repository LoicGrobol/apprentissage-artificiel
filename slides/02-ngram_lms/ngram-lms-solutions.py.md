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

On peut évidemment copier-coller le code au-dessus, mais on peut aussi réutiliser ce qu'on a déjà défini :

```python
def crude_tokenizer_and_normalizer(s):
    return crude_tokenizer(s.lower())

asser = crude_tokenizer_and_normalizer("Je reconnais l'existence du kiwi-fruit.") == [
    'je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit'
]
```

## 💜 Extraire les bigrammes 💜

Écrire une fonction `extract_bigrams` qui prend en entrée une liste de mots et renvoie la liste des bigrammes correspondants sous forme de couples de mots.


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


On va ensuite estimer les probabilités de transition, c'est-à-dire la probabilité de générer un
certain mot $w_1$ sachant que le mot précédent est $w_0$. On le fait en utilisant la formule du
maximum de vraisemblance :

\begin{equation}
   P(w_1|w_0) := P\!\left([w_0, w_1]~|~[w_0, *]\right) = \frac{\text{nombre d'occurrences du bigramme $w_0 w_1$}}{\text{nombre d'occurrences de l'unigramme $w_0$}}
\end{equation}

Pour que ce soit plus agréable à sampler on va utiliser un dictionnaire de dictionnaires :
`probs[v][w]` stockera $P(w|v)$.

À vous de jouer : écrire une fonction `get_probs`, qui prend en entrée la les compteurs de bigrammes
et d'unigrammes et renvoie le dictionnaire `probs`

```python
def get_probs(unigram_counts, bigram_counts):
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

## 🤔 Générer 🤔

Pour l'instant on ne va pas se préoccuper de sauvegarder le modèle on va l'utiliser directement pour
sampler. Le principe est simple : on choisit le premier mot, puis on choisit le deuxième mot en
prenant en compte celui qu'on vient de générer (le premier donc si vous suivez) et ainsi de suite.


**Questions**

- Comment on choisit le premier mot ?
- Et quand est-ce qu'on décide de s'arrêter ?


Jurafsky et Martin nous disent

>  We’ll first need to augment each sentence with a special symbol `<s>` at the beginning of the
> sentence, to give us the bigram context of the first word. We’ll also need a special end-symbol.
> `</s>`

Heureusement on a un fichier bien fait : il y a une seule phrase par ligne.


1\. Modifier `read_corpus` pour ajouter ajouter à la volée `<s>` au début de chaque ligne et `</s>` à la fin de chaque ligne.

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

Il y a encore un petit problème

```python
bigram_counts.most_common(1)
```

🤔


On a compté les lignes vides 😤. Ça ne posait pas de problème jusque-là puisque ça n'ajoutait rien
aux compteurs de n-grammes, mais maintenant ça nous fait des `["<s>", "</s>"]`.


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

Voici par exemple comment choisir un mot qui suivrait « je » :

```python
# Les candidats mots qui peuvent suivre « je »
candidates = list(probs["je"].keys())
# Leurs poids, ce sont les probabilités qu'on a déjà calculé
weights = [probs["je"][c] for c in candidates] 
random.choices(candidates, weights, k=1)[0]  # Attention `choices` renvoit une liste
```

Écrire une fonction `sample` qui prend en argument les probabilités de bigrammes (sous la forme d'un dictionnaire de dictionnaires comme notre `prob`) et génère une phrase en partant de `<s>` et en ajoutant des mots itérativement, s'arrêtant quand `</s>` a été choisi.

```python
def sample(bigram_probs):
    sent = ["<s>"]
    while sent[-1] != "</s>":
        candidates = list(probs[sent[-1]].keys())
        weights = [probs[sent[-1]][c] for c in candidates]
        sent.append(random.choices(candidates, weights)[0])
    return sent
```

Pas de assert ici comme on a de l'aléatoire, mais la cellule suivante permet de tester si ça marche

```python
print(sample(probs))
print(" ".join(sample(probs)[1:-1]))
```

C'est rigolo, hein ?


Qu'est-ce que vous pensez des textes qu'on génère ?

## 🧐 Aller plus loin 🧐


En vous inspirant de ce qui a été fait, coder un générateur de phrases à partir de trigrammes,
tétragrammes (4), puis de n-grammes arbitraires.

