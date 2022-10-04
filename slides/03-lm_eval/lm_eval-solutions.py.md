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

<!-- #region slideshow={"slide_type": "slide"} -->
Évaluer les modèles de langue à n-grammes : corrections
===================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-10-05
<!-- #endregion -->

```python
import math
import random
import re
from collections import Counter, defaultdict
```

## Précédently

```python
def crude_tokenizer_and_normalizer(s):
    tokenizer_re = re.compile(
        r"""
        (?:                   # Dans ce groupe, on détecte les mots
            \b\w+?\b          # Un mot c'est des caractères du groupe \w, entre deux frontières de mot
            (?:               # Éventuellement suivi de
                '             # Une apostrophe
            |
                (?:-\w+?\b)*  # Ou d'autres mots, séparés par des traits d'union
            )?
        )
        |\S        # Si on a pas détecté de mot, on veut bien attraper un truc ici sera forcément une ponctuation
        """,
        re.VERBOSE,
    )
    return tokenizer_re.findall(s.lower())

crude_tokenizer_and_normalizer("La lune et les Pléiades sont déjà couchées : la nuit a fourni la moitié de sa carrière, et moi, malheureuse, je suis seule dans mon lit, accablée sous le chagrin.")
```

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
    for (*previous_words, target_word), c in ngram_counts.items():
        probs[tuple(previous_words)][target_word] = c/unigram_counts[target_word]

    return dict(probs)
```

```python
def generate_from_ngrams(ngram_probs):
    # J'avais dit qu'on pouvait. Ici on le fait salement
    n = len(next(iter(ngram_probs.keys()))) + 1
    sent = ["<s>"] * (n-1)
    while sent[-1] != "</s>":
        # Essayer de bien réfléchir pour comprendre le `1-n`
        previous_words = tuple(sent[1-n:])
        candidates = list(ngram_probs[previous_words].keys())
        weights = [ngram_probs[previous_words][c] for c in candidates]
        sent.append(random.choices(candidates, weights)[0])
    # Pas la peine de renvoyer les tokens <s>
    return sent[n-1:-1]
```

```python
# Unpacking dans un appel de fonction, on va voir qui ? Oui, la doc.
bigram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 2))

for _ in range(8):
    print(" ".join(generate_from_ngrams(bigram_probs)))
```

```python
# Unpacking dans un appel de fonction, on va voir qui ? Oui, la doc.
trigram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 3))

for _ in range(8):
    print(" ".join(generate_from_ngrams(trigram_probs)))
```

```python
pentagram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 5))

for _ in range(8):
    print(" ".join(generate_from_ngrams(pentagram_probs)))
```

## 🎲 Vraisemblance d'une phrase 🎲


Écrire une fonction `sent_likelihood`, qui prend en argument des probas de n-grammes, une phrase
sous forme de liste de chaînes de caractères (et éventuellement `n` si vous ne voulez pas utiliser
mon astuce sale du début) et renvoie la vraisemblance de cette phrase.


```python
def sent_likelihood(ngram_probs, sent, n):
    sent = ["<s>"] * (n-1) + sent + ["</s>"]
    total_likelihood = 1.0
    for i in range(len(sent)-n+1):
        total_likelihood = total_likelihood * ngram_probs[tuple(sent[i:i+n-1])][sent[i+n-1]]
    return total_likelihood

assert sent_likelihood(trigram_probs, ["pénitentes", ",", "que", "prenez-vous", "?"], 3) == 3.9225257586711874e-14
```

Avec de la syntaxe plus Pythonique

```python
def sent_likelihood(ngram_probs, sent, n):
    # Unpacking, encore et toujours
    sent = [*(["<s>"] * (n-1)), *sent, "</s>"]
    total_likelihood = 1.0
    for i in range(len(sent)-n+1):
        total_likelihood *= ngram_probs[tuple(sent[i:i+n-1])][sent[i+n-1]]
    return total_likelihood

assert sent_likelihood(trigram_probs, ["pénitentes", ",", "que", "prenez-vous", "?"], 3) == 3.9225257586711874e-14
```

## 🤘🏻 Calculer la perplexité 🤘🏻

1\. Écrire une fonction `sent_loglikelihood`, qui les mêmes arguments que `sent_likelihood` et
renvoie la **log-vraisemblance** de cette phrase.


```python
def sent_loglikelihood(ngram_probs, sent, n):
    sent = ["<s>"] * (n-1) + sent + ["</s>"]
    result = 0.0
    for i in range(len(sent)-n+1):
        # Notez que du coup, on pourrait dès le début calculer les log-probas de n-grammes
        result += math.log(ngram_probs[tuple(sent[i:i+n-1])][sent[i+n-1]])
    return result

assert sent_loglikelihood(trigram_probs, ["pénitentes", ",", "que", "prenez-vous", "?"], 3) == -30.86945552941164
```


2\. Écrire une fonction `avg_log_likelihood`, qui prend qui prend en argument des probas de
n-grammes, un chemin vers un corpus et `n` et renvoie la vraisemblance moyenne de ce corpus estimée
par le modèle à n-grammes correspondant à ces probas (autrement dit, la moyenne des
log-vraisemblances de phrases). Testez sur *Le Ventre de Paris* (non, on l'a dit, c'est pas une
évaluation juste, mais ça va nous permettre de voir si ça marche).

```python
def avg_log_likelihood(ngram_probs, file_path, n):
    log_likelihood_sum = 0.0
    n_sents = 0
    with open(file_path) as in_stream:
        for line in in_stream:
            words = crude_tokenizer_and_normalizer(line.strip())
            log_likelihood_sum += sent_loglikelihood(ngram_probs, words, n)
            n_sents += 1
    
    return log_likelihood_sum/n_sents

assert avg_log_likelihood(bigram_probs, "data/zola_ventre-de-paris.txt", 2) == -56.321217776181875
assert avg_log_likelihood(trigram_probs, "data/zola_ventre-de-paris.txt", 3) == -81.20968449380536
assert avg_log_likelihood(pentagram_probs, "data/zola_ventre-de-paris.txt", 5) == -88.25016939038316
```

3\. En pratique, on évalue pas vraiment les modèles de langues avec les log-vraisemblances, mais
avec une quantité appelée **perplexité**, définie comme $\exp(-\text{avg-log-l})$, où
$\text{avg-log-l}$ est la log-vraisemblance moyenne et $\exp$ est la fonction exponentielle
`math.exp`. Écrire une fonction `perplexity` qui calcule la perplexité d'un modèle de langue à
n-grammes sur un corpus donné. Là encore, testez avec les modèles précédents et *Le Ventre de
Paris*.

```python
def perplexity(ngram_probs, file_path, n):
    return math.exp(-avg_log_likelihood(ngram_probs, file_path, n))
```
