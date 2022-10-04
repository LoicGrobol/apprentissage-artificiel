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
Cours 3 : Évaluer et améliorer les modèles de langue à n-grammes
=====================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-10-04
<!-- #endregion -->

## Précédently

```python
import random
import re
from collections import Counter, defaultdict
```

Rappel de [l'épisode précédent](../02-ngram_lms/ngram-lms-slides.py.md) : on a vu comment utiliser
des fréquences de n-grammes issues d'un corpus pour générer du texte :


(Cette fois-ci on va aussi garder la ponctuation, et comme l'expression régulière devient compliqué,
on va y mettre [des commentaires](https://docs.python.org/fr/3/library/re.html#re.VERBOSE).)

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

Par exemple pour des bigrammes, voici ce qu'on génère :

```python
# Unpacking dans un appel de fonction, on va voir qui ? Oui, la doc.
bigram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 2))

for _ in range(8):
    print(" ".join(generate_from_ngrams(bigram_probs)))
```

C'est pas *affreux* au sens où ça ressemble bien à du langage, mais c'est pas *super* parce qu'il y
a quand même plein de défauts. Ce n'est pas très surprenant : comme on l'a vu la dernière fois,
c'est un modèle très simpliste.


Les choses s'arrangent un peu si on prend un $n$ plus grand, par exemple $3$ :

```python
# Unpacking dans un appel de fonction, on va voir qui ? Oui, la doc.
bigram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 3))

for _ in range(8):
    print(" ".join(generate_from_ngrams(bigram_probs)))
```

Alors pourquoi pas essayer avec plus ? Par exemple $5$ ?

```python
pentagram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 5))

for _ in range(8):
    print(" ".join(generate_from_ngrams(pentagram_probs)))
```

Ça a l'air bien ! Elles sont propres ces phrases ! Par exemple sur une exécution de la cellule
précédente, j'ai obtenu `"alors seulement lisa rougit de se voir à côté de ce garçon ."`

```python
!grep "Alors seulement Lisa rougit" "data/zola_ventre-de-paris.txt"
```

Oups

## Les plaies de l'apprentissage


On verra plus tard dans le cours une définition formelle de l'apprentissage automatique/artificiel,
mais ce qu'on a ici c'en est : on a conçu un programme qui ingère des données et essaie de produire
des résultats qui reproduisent (avec des gros guillemets) ce qu'il y avait dans ces données. On dit
qu'il a **appris** ces données, dont ont dit qu'on appelle **données d'entraînement**.


Alors ici quel est le problème ? On voit que pour $n=5$, le modèle de langue à $n$-grammes commence
à avoir un comportement désagréable : il devient essentiellement une machine à apprendre par cœur le
corpus d'entraînement. Ça n'est pas un modèle qui est *incorrect* : il remplit bien le cahier des
charges. Par contre, c'est un modèle complètement *inintéressant* : à ce compte-là, autant
directement tirer au sort une phrase dans le corpus.

Autrement dit, on a un problème parce que d'une certaine façon, le modèle a trop bien appris : on
dit qu'on est en situation de **sur-apprentissage** (*overfitting*).


Cela dit, on a vu que ne pas assez apprendre ce n'était pas non plus super : par exemple le modèle à
bigrammes n'est pas très satisfaisant non plus. Lui semblait au contraire ne pas être assez riche
pour apprendre correctement ce qu'il y avait dans les données d'entraînement. On parle dans ce cas
de **sous-apprentissage** (*underfitting*).

Celui qui avait l'air d'être le meilleur équilibre, en fait, c'était le modèle à trigrammes.

Mais ça, c'est juste notre intuition, ce qui est bien, mais limité. En particulier, c'est loin
d'être infaillible. Ce qu'il nous faudrait — comme Newton ou Marie Skłodowska–Curie — c'est une
façon de le **quantifier**.


Autrement dit : on veut moyen, étant donné un modèle de langue, de déterminer sa qualité sous forme
d'un **score** chiffré (ou **métrique**). On dit aussi qu'on veut pouvoir **évaluer** ce modèle.

Pourquoi un score chiffré ? Pourquoi pas juste un truc qui dirait « bien » ou « pas bien » ? Pour
deux raisons :

- La raison principale, c'est que ça permet de **comparer** des modèles. On ne veut pas seulement
  savoir si un modèle est bon : on veut savoir lequel est le *meilleur*
- L'autre raison, c'est que si on fait bien les choses, on peut attribuer des **interprétations**
  aux scores obtenus.
  - Le cas prototypique, ce serait un score qui serait sur une échelle de $0$ à $1$, où
    - $0$ serait un modèle qui se trompe tout le temps
    - $1$ un modèle qui ne se trompe jamais
    - $0.5$ un modèle qui se trompe une fois sur deux.
    - etc.
  - On parle de score **calibré** et/ou **interprétable**.

## Comment on fait un score

Il y a un moyen très très simple d'évaluer un modèle de langue (ou en fait à peu près n'importe quel
modèle de TAL) :

- Attrapez un⋅e humain⋅e
- Expliquez-lui la tâche (ici générer du texte qui soit crédible **et** original)
- Montrez-lui les sorties du système
- Demandez-lui de mettre une note ua système

Ça pose plusieurs problèmes, évidemment.


Une solution qui permet d'améliorer un peu la régularité de la procédure (par exemple pour éviter
que la note ne soit donnée que sur la petite partie des phrases dont votre sujet arrive à se
souvenir), c'est de demander de donner une note *par sortie (donc ici une note par phrase) et de
faire une moyenne.

C'est pas encore parfait, mais c'est déjà un peu mieux.

Évidemment ça ne résout pas la question de la praticité : trouver des humain⋅es disposé⋅es à faire
ça, c'est pas évident, c'est lent, ça coûte cher…


Notre objectif ici, ça va donc être — comme souvent — de trouver une façon d'automatiser ceci.


(Notez qu'en termes de recherche, comment évaluer la génération de texte, c'est vraiment une
question ouverte. Si la question vous intéresse, [la thèse d'E.
Manning](https://esmanning.github.io/projects/project_diss/) peut être un point de départ
intéressant.)
