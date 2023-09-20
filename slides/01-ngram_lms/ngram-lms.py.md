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

<!-- #region slideshow={"slide_type": "slide"} -->
TP 1 : Modèles de langues à n-grammes
========================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

## Modèles de langues

Qu'est-ce que vous pensez des phrases suivantes ?

> Bonjour, ça va ?


> Je reconnais l'existence du kiwi-fruit.


> Les idées vertes incolores dorment furieusement.


> Vous désastre réjouirez de que ce aucun.


> oijj eofiz ipjij paihefoîozenui.


Est-ce qu'il y en a qui vous parlent plus que d'autres ? Pourquoi ?


Pour plein de raisons, étant donné un langage (et une variété de ce langage, etc.), il y a des
phrases qu'on risque de voir ou d'entendre plus souvent que d'autres.


On peut dire ainsi que certaines phrases sont plus **vraisemblables** que d'autres.


On peut y penser de la manière suivante (pour l'instant) :

- On prend toutes les phrases qui ont été un jour prononcées dans cette langue.
- On les écrit toutes (avec répétition) sur des bouts de papiers.
- On met les bouts de papier dans une urne géante, on touille et on en choisit un.


On peut alors parler de *probabilité* d'avoir choisi une phrase donnée. Et se demander :

> Si j'ai une phrase, par exemple « Toi dont le trône étincelle, ô immortelle Aphrodite. », comment
> estimer cette probabilité ?


Un modèle de langue, c'est un **modèle** qui permet d'**estimer** la **vraisemblance** d'une
**phrase**.


Notre objectif aujourd'hui c'est de voir comment on fait ça, d'abord en théorie, puis en pratique
sur une application marrante et très très très à la mode : la génération de textes.


À quoi ça sert ?


À plein de trucs

- Traduction automatique :
  - $P(\text{moche temps pour la saison}) > P(\text{sale temps pour la saison})$
- Correction orthographique :
  - Je ne peux pas **croitre** cette histoire
  - $P(\text{peux pas croire cette}) > P(\text{peux pas croitre cette})$
- Reconnaissance de la parole (ASR)
  - $P(\text{Par les temps qui courent}) ≫ P(\text{Parle et t'en qui cours})$
- Résumé automatique, questions/réponses…


On se basera pour la théorie et les notations sur le chapitre 3 de [*Speech and Language
Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et James H. Martin. À ta
place, je le garderais donc à portée de main, le poly *et* les slides (et je prendrais le temps de
lire les chapitres précédents au calme).

## Formalisons (un peu)


On veut assigner des probabilités (≈) à des séquences de mots.


Si on note une séquence de mots $S = w_1, w_2, …, w_n$, on notera sa probabilité $P(w_1, w_2, …,
w_n)$.


### Estimateur du maximum de vraisemblance

Rappel : on peut estimer la probabilité d'un truc en calculant sa fréquence d'apparition.


Par exemple, si on veut estimer la probabilité qu'un dé truqué fasse 6 :

- On lance le dé un grand nombre de fois (mettons qu'on choisisse 1000), on parle d'**échantillon**.
- On compte le nombre de fois qu'on a obtenu 6, imaginons que c'est 271.
- On calcule la **fréquence d'apparition** de 6 : \frac{271}{1000} = 0.271.
- On **choisit** cette valeur comme estimation de la probabilité d'avoir 6


Notez que c'est bien une estimation, et qu'elle n'est pas infaillible. On peut obtenir 1000 fois 6
de suite, même avec un dé équilibré. C'est improbable, mais ça peut arriver, et dans ce cas notre
estimation de la probabilité sera affreusement fausse.


Cette façon d'estimer une probabilité, c'est (un cas particulier de) l'**estimateur du maximum de
vraisemblance**. La façon la plus simple d'estimer des probabilités.


Ok, super, il donne quoi cet estimateur pour notre problème ? En quoi ça consiste ? À votre avis ?


Et bien imaginons qu'on veuille déterminer la probabilité d'une phrase, par exemple « le petit chat
est content ».

- On prend un gros corpus (c'est notre échantillon).
- On regarde combien de fois cette phrase apparaît.
- Et on divise par la taille du corpus.


Voyons ce que ça donne :

- [Combien de pages sur Google pour cette
  requête](https://www.google.com/search?q=%22le+petit+chat+est+content%22).
- Combien de pages au total dans l'index de Google ? Dur à savoir, mais probablement de l'ordre de
  grandeur de $100 000 000 000$.

On estimerait alors la probabilité de cette phrase à $0.00000000008$.


Ok, parfait, on a fini ?


C'est quoi la probabilité de « je reconnais l'existence du kiwi-fruit » alors ?


<https://www.google.com/search?q=%22je+reconnais+l'existence+du+kiwi-fruit%22>


Alors ?


$0$ ?


Mais « Vous désastre réjouirez de que ce aucun ». Ça serait zéro aussi alors ? Est-ce que vraiment
on veut mettre la même probabilité à ces deux phrases ?


Oups.


Le problème, c'est que l'échantillon qu'il nous faudrait ce n'est pas un échantillon de tout ce qui
a déjà été produit comme phrase, mais un échantillon de tout ce qui **pourrait** être produit. Et
évidemment ce n'est pas accessible.

### Décomposer pour régner


Ok, [essayons encore](https://www.youtube.com/watch?v=Xg4Pa3DORCE).


Il nous faut une façon plus subtile de procéder. On va se reposer pour ça sur une propriété
intéressante du langage humain :


Si je dis : « je suis en train d'écrire sur le… ». Quel est le mot suivant d'après-vous ?


Il y a évidemment plusieurs solutions. Mais *certaines semblent plus vraisemblables*. 🧐.


Autrement dit : il y a une corrélation (attention, pas un conditionnement total) imposée par le
début d'une phrase sur sa suite.


On va s'appuyer sur ça pour proposer un modèle de langue qui soit **implémentable** (et après ~~on~~
vous allez l'implémenter).


On va imaginer un modèle de langue qui fonctionne comme un **processus aléatoire**, c'est-à-dire
comme une série de décisions aléatoires. En l'occurrence, on va imaginer un processus où la phrase
est générée mot par mot.


Autrement dit :

- On choisit le premier mot $w_0$ en regardant pour un corpus échantillon les fréquences des mots
  apparaissant en début de phrase.
- On choisit le deuxième mot $w_1$ en regardant les fréquences des mots apparaissant en deuxième
  position dans les phrases qui commencent par $w_0$.
- On choisit $w_2$ en regardant les mots qui apparaissent en troisième position dans les phrases qui
  commencent par $w_0, w_1$
- …

<!-- TODO: commencer par regarder le calcul des probas sur un petit exemple comme dans J&M avant de passer au cas général -->


Les probabilités ici sont plus faciles à estimer :

La probabilité $P([w_0, *])$ (qu'on notera aussi $P(w_0)$) qu'un mot apparaisse en début de phrase,
c'est

\begin{equation}
    P(w_0) = \frac{\text{Nombre de phrases qui commencent par $w_0$}}{\text{Nombre de phrases dans le corpus}}
\end{equation}


La probabilité $P([w_0, w_1, *]~|~[w_0, *])$, ou $P(w_1|w_0)$ qu'une phrase commence par $w_0, w_1$
sachant qu'elle commence par $w_1$ (on parle de probabilité conditionnelle), c'est

\begin{equation}
    P(w_1|w_0) = \frac{\text{Nombre de phrases qui commencent par $w_0, w_1$}}{\text{Nombre de phrases qui commencent par $w_0$}}
\end{equation}

et ainsi de suite.


Et c'est quoi alors la probabilité de la phrase entière ? Et bien, c'est simplement le produit des
probabilités, comme quand on suit une série d'expériences avec un arbre :

\begin{equation}
    P(w_0, w_1, …, w_n) = P(w_0) × P(w_1|w_0) × P(w_2|w_0, w1) × … × P(w_n | w_0, w_1, …, w_{n-1})
\end{equation}

### N-grammes

Évidemment ça ne pouvait pas être si simple.


**Évidemment.**


Le problème ici, c'est que la procédure itérative qu'on a décrite marche bien en début de phrase,
mais en fin de phrase on retombe sur le problème précédent.

\begin{equation}
    P(\text{vert}~|~\text{Je}, \text{reconnais}, \text{l'}, \text{existence}, \text{du}, \text{kiwi-fruit})
\end{equation}


On va donc faire une hypothèse un peu grossière : on va supposer par exemple que

\begin{equation}
    P(w_3~|~w_0, w_1, w_2) = P(w_3~|~w_2)
\end{equation}

Autrement dit la probabilité d'apparition d'un mot ne dépend que des $n-1$ (ici $1$) mots
précédents. Nous donnant ainsi un **modèle de langue à n-grams** (ici bigrammes).

## À vous de jouer !

Notre objectif ici sera de faire de la **génération de textes**.

Pour les données on va d'abord travailler avec [Le Ventre de
Paris](../../data/zola_ventre-de-paris.txt) qui est déjà dans ce repo pour les tests puis avec [le
corpus CIDRE](https://www.ortolang.fr/market/corpora/cidre) pour passer à l'échelle, mais on
pourrait aussi utiliser Wikipedia (par exemple en utilisant
[WikiExtractor](https://github.com/attardi/wikiextractor)) ou [OSCAR](https://oscar-corpus.com/).

On va devoir faire les choses suivantes (pour un modèle à bigrammes)

- Extraire les unigrammes et les bigrammes d'un corpus
- Calculer les probas normalisées des bigrammes
- Sampler des phrases à partir du modèle

On va essayer de faire les choses à la main, sans trop utiliser de bibliothèques, pour bien
comprendre ce qui se passe.

Puis on étendra à des trigrammes et des n-grammes.

## ✂️ Tokenization ✂️

1\. Écrire une fonction `crude_tokenizer` qui prend comme argument une chaine de caractères et
    renvoie la liste des mots de cette chaîne en séparant sur les espaces.

```python tags=["raises-exception"]
def crude_tokenizer(s):
    pass # À toi de coder

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', "l'existence", 'du', 'kiwi-fruit.'
]
```

2\. Modifier la fonction `crude_tokenizer` pour qu'elle sépare aussi suivant les caractères
   non alphanumériques. **Indice** ça peut être utile de revoir [la doc sur les expressions
   régulières](https://docs.python.org/3/library/re.html) ou de relire [un tuto à ce
   sujet](https://realpython.com/regex-python/).

```python tags=["raises-exception"]
def crude_tokenizer(s):
    pass # À toi de coder

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', 'l', 'existence', 'du', 'kiwi', 'fruit'
]
```

3\. On aimerait maintenant garder les apostrophes à la fin du mot qui les précède, ainsi que les
mots composés ensemble.

```python tags=["raises-exception"]
def crude_tokenizer(s):
    pass # À toi de coder

assert crude_tokenizer("Je reconnais l'existence du kiwi-fruit.") == [
    'Je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit'
]
```

4\. Écrire une fonction `crude_tokenizer_and_normalizer` qui en plus de tokenizer comme précédemment
met tous les mots en minuscules

On peut évidemment copier-coller le code au-dessus, mais on peut aussi réutiliser ce qu'on a déjà
défini :

```python tags=["raises-exception"]
def crude_tokenizer_and_normalizer(s):
    pass # À toi de coder

asser = crude_tokenizer_and_normalizer("Je reconnais l'existence du kiwi-fruit.") == [
    'je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit'
]
```

## 💜 Extraire les bigrammes 💜

Écrire une fonction `extract_bigrams` qui prend en entrée une liste de mots et renvoie la liste des
bigrammes correspondants sous forme de couples de mots.


Version directe

```python tags=["raises-exception"]
def extract_bigrams(words):
    pass # À toi de coder

assert extract_bigrams(['je', 'reconnais', "l'", 'existence', 'du', 'kiwi-fruit']) == [
    ('je', 'reconnais'),
     ('reconnais', "l'"),
     ("l'", 'existence'),
     ('existence', 'du'),
     ('du', 'kiwi-fruit')
]
```


## 🔢 Compter 🔢


Écrire une fonction `read_corpus` qui prend en argument un chemin vers un fichier texte, l'ouvre, le
tokenize et y compte les unigrammes et les bigrammes en renvoyant deux `Counter` associant
respectivement à chaque mot et à chaque bigramme leurs nombres d'occurrences.

```python tags=["raises-exception"]
from collections import Counter
    
def read_corpus(file_path):
    unigrams = Counter()
    bigrams = Counter()
    pass # À toi de coder
    
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
   P(w_1|w_0) := P\!\left([w_0, w_1]~|~[w_0, *]\right)
    = \frac{
        \text{nombre d'occurrences du bigramme $w_0 w_1$}
      }{
        \text{nombre d'occurrences de l'unigramme $w_0$}}
\end{equation}

Pour que ce soit plus agréable à sampler on va utiliser un dictionnaire de dictionnaires :
`probs[v][w]` stockera $P(w|v)$.

À vous de jouer : écrire une fonction `get_probs`, qui prend en entrée les compteurs de bigrammes
et d'unigrammes et renvoie le dictionnaire `probs`.

```python tags=["raises-exception"]
def get_probs(unigram_counts, bigram_counts):
    pass # À toi de coder

probs = get_probs(unigram_counts, bigram_counts)
assert probs["je"]["déjeune"] == 0.002232142857142857
```

**Astuce** on peut utilise un `defaultdict`.

## 💁🏻 Générer un mot 💁🏻

**Bon c'est bon maintenant ?**


Oui ! On va enfin pouvoir générer des trucs !


Pour ça on va piocher dans le module [`random`](https://docs.python.org/3/library/random.html) de la
bibliothèque standard, et en particulier la fonction
[`random.choices`](https://docs.python.org/3/library/random.html#random.choices) qui permet de tirer
au sort dans une population finie en précisant les probabilités (ou *poids*) de chacun des éléments.
Les poids n'ont en principe pas besoin d'être normalisés (mais ils le seront ici, vu comme on les a
construits).

```python
import random
```

Voici par exemple comment choisir un élément dans la liste `["a", "b", "c"]` en donnant comme
probabilités respectives à ses éléments $0.5$, $0.25$ et $0.25$

```python
candidates = ["a", "b", "c"]
weights = [0.5, 0.25, 0.25]
random.choices(candidates, weights, k=1)[0]  # Attention: `choices` renvoit une liste
```

À vous de jouer : écrire une fonction `gen_next_word` qui prend en entrée le dictionnaire `probs` et
un mot et renvoie en sortie un mot suivant, choisi en suivant les probabilités estimées précédemment


## 🤔 Générer un texte  🤔

On va maintenant pouvoir utiliser notre modèle pour générer du texte. Le principe est simple : on
choisit le premier mot, puis on choisit le deuxième mot en prenant en compte celui qu'on vient de
générer (le premier donc si vous suivez) et ainsi de suite.


**Questions**

- Comment on choisit le premier mot ?
- Et quand est-ce qu'on décide de s'arrêter ?


Jurafsky et Martin nous disent

<!-- LTeX: language=en-Us -->
> We’ll first need to augment each sentence with a special symbol `<s>` at the beginning of the
> sentence, to give us the bigram context of the first word. We’ll also need a special end-symbol.
> `</s>`
<!-- LTeX: language=fr -->

Heureusement on a un fichier bien fait : il y a une seule phrase par ligne.


1\. Modifier `read_corpus` pour ajouter à la volée `<s>` au début de chaque ligne et `</s>` à la fin
de chaque ligne.

```python tags=["raises-exception"]
def read_corpus(file_path):
    pass # À toi de coder
    
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

```python tags=["raises-exception"]
bigram_counts.most_common(1)
```

🤔


On a compté les lignes vides 😤. Ça ne posait pas de problème jusque-là puisque ça n'ajoutait rien
aux compteurs de n-grammes, mais maintenant ça nous fait des `["<s>", "</s>"]`.


2\. Modifier `read_corpus` pour ignorer les lignes vides

```python tags=["raises-exception"]
def read_corpus(file_path):
    pass # À toi de coder


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

```python tags=["raises-exception"]
def generate(bigram_probs):
    pass # À toi de coder
```

Pas de `assert` ici comme on a de l'aléatoire, mais la cellule suivante permet de tester si ça
marche :

```python
print(generate(probs))
```

Et ici pour avoir du texte qui ressemble à quelque chose :

```python tags=["raises-exception"]
print(" ".join(generate(probs)[1:-1]))
```

C'est rigolo, hein ?


Qu'est-ce que vous pensez des textes qu'on génère ?

## 🧐 Aller plus loin 🧐


En vous inspirant de ce qui a été fait, coder un générateur de phrases à partir de trigrammes, puis
de n-grammes arbitraires.
