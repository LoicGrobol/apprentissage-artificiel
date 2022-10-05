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
Cours 3 : Évaluer les modèles de langue à n-grammes
===================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2022-10-05
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
trigram_probs = get_ngram_probs(*read_corpus_for_ngrams("data/zola_ventre-de-paris.txt", 3))

for _ in range(8):
    print(" ".join(generate_from_ngrams(trigram_probs)))
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
qu'il a **appris** ces données, qu'on appelle **données d'entraînement**.


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

## Comment l'on fabrique un score

### Une astuce surprenante pour évaluer des modèles

Il y a un moyen très très simple d'évaluer un modèle de langue (ou en fait à peu près n'importe quel
modèle de TAL) :

- Attrapez un⋅e humain⋅e
- Expliquez-lui la tâche (ici générer du texte qui soit crédible **et** original)
- Montrez-lui les sorties du système
- Demandez-lui de mettre une note au système

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

### Une solution de secours

Essayons de nous rappeler ce qu'on disait [pas plus tard que la semaine
dernière](../02-ngram_lms/ngram-lms-slides.py.md)


> Un modèle de langue, c'est un **modèle** qui permet d'**estimer** la **vraisemblance** d'une
> **phrase**.


On avait dit que dans un monde idéal où on disposerait d'un corpus $C$ de toutes les phrases pouvant
apparaître dans la langue qu'on étudie.


En mettant de côté le fait que certaines phrases sont plus courantes que d'autres, ce qu'on pourrait
attendre d'un modèle parfait, c'est qu'il donne une vraisemblance de $1$ pour toutes les phrases de
ce corpus (puisqu'elles sont toutes possibles) et une vraisemblance de $0$ pour toutes les phrases
qui n'y sont pas (puisqu'elles ne sont pas possibles).


Un modèle qui donnerait une vraisemblance $p_s = 0.1$ pour une phrase $s$ qui serait dans $C$ (on
note $s∈C$), il aurait par exemple un score pas super pour cette phrase. Autrement dit, on peut
simplement utiliser la vraisemblance donnée par le modèle comme une métrique d'évaluation !


Bon, ça c'est sur une phrase, mais on peut facilement l'étendre à l'ensemble du corpus, en faisant
une moyenne :


Si on imagine qu'il y a $n$ phrases possibles dans $C$ (un très très grand $n$ donc) et qu'on les
appelle $s_1$, $s_2$, …, $s_n$, le score global $M$ de notre modèle, on peut par exemple décider que
ce serait la moyenne :


\begin{equation}
    M = \frac{p_{s_1} + p_{s_2} + … + p_{s_n}}{n}
\end{equation}



**Attention**, j'insiste : on *peut* **décider** de **choisir** ceci comme score. Il n'y a pas de
divinité des modèles de langues qui aurait décidé que ce serait la bonne façon. Il n'y pas de
métrique d'évaluation canonique et parfaite qui serait écrite dans la trame de l'univers.

Les métriques d'évaluations, scores, mesures de qualités… ce sont des choses créées et choisies par
les humains. Souvent avec de bonnes raisons de faire ces choix, mais ça reste bien des **choix**. On
pourrait en faire d'autre, et dans ce cas précis, comme je l'ai dit plus haut, il n'y a actuellement
pas de consensus scientifique sur comment évaluer un modèle de génération de texte de façon
*satisfaisante*.


Ici par exemple, vous remarquerez que ce score ne tient pas compte de la vraisemblance donnée aux
phrases impossibles (celles qui ne seraient pas dans $C$). Pour des raisons pratiques : il y en a
une infinité ! Pas facile de faire une moyenne. On se contente donc du choix fait ici : la moyenne
des vraisemblances des phrases de $C$.


Ici on a donc juste un exemple de score. Et on va de toute façon vite tomber sur un écueil.


**Vous voyez le problème ?**


On a **toujours** pas de « corpus de toutes les phrases pouvant apparaître dans la langue qu'on
étudie »


Comment on fait alors ?


Et bien comme d'habitude : on prend un échantillon. Un corpus, quoi. On va essayer de le prendre le
plus représentatif possible de la langue. C'est le mieux qu'on puisse faire de toute façon.


Et tout le monde est content. Bon il faut se souvenir de comment on obtient $p_s$, mais on va
vite le revoir.


### Corpus d'évaluation


Ça tombe bien, on a déjà sous la main un échantillon de la langue : notre corpus d'entraînement !


Du coup on a qu'à l'utiliser pour évaluer notre modèle, non ?


Quoi ? Encore un problème ?


Ben oui, souvenez-vous du modèle 5-gramme


Le problème, c'est que si on fait- ça, un modèle qui apprend juste le corpus par cœur va obtenir un
super score, puisqu'il a eu accès à ces données-là pour apprendre.


Alors pourquoi pas, mais imaginez que vous ayez deux modèles. Un appris en utilisant *Frankenstein*,
l'autre en utilisant *Le Ventre de Paris*. Si vous les évaluez en regardant s'ils prédisent
correctement *Le Ventre de Paris*, il y en a un qui a un avantage déloyal. Autrement dit on a une
évaluation qui ne nous dira pas vraiment lequel de ces modèles aurait le meilleur score dans
l'absolu, dans le monde idéal où on pourrait le tester sur ce corpus magique qui contient toute la
langue.


Pour que ce soit équitable, on va donc plutôt préférer utiliser comme échantillon pour l'évaluation
un corpus différent du corpus d'entraînement, si possible même un qui n'a aucune phrase en commun.
On appellera ce deuxième corpus le **corpus de test** ou d'**évaluation**.


## Perplexité

### 🎲 Vraisemblance d'une phrase 🎲


Rappel : un modèle de langue à trigrammes calcule la vraisemblance d'une phrase $s= w_1, …, w_n$ de
la façon suivante :

\begin{equation}
    p_s = P(w_0) × P(w_1 | w_0) × P(w2 | w_0, w_1) × P(w_3 | w_1, w_2) × … × P(w_n | w_{n-2}, w_{n-1})
\end{equation}

avec


\begin{equation}
    P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{Fréquence du trigramme $w_{i-2}, w_{i-1}, w_i$}}{\text{Fréquence de l'unigramme $w_i$}}
\end{equation}


Et pour $P(w_0)$ $P(w_1 | w_0)$, on peut tricher en les écrivant $P(w_0 | \text{<s>}, \text{<s>})$
et $P(w_1 | \text{<s>}, w_0)$. Ce serait aussi bien de compter que $w_n = \text{</s>}$.


Écrire une fonction `sent_likelihood`, qui prend en argument des probas de n-grammes, une phrase
sous forme de liste de chaînes de caractères (et éventuellement `n` si vous ne voulez pas utiliser
mon astuce sale du début) et renvoie la vraisemblance de cette phrase.



```python
def sent_likelihood(ngram_probs, sent, n):
    pass  # À toi de coder

assert sent_likelihood(trigram_probs, ["pénitentes", ",", "que", "prenez-vous", "?"], 3) == 3.9225257586711874e-14
```

### Log-vraisemblance


Il y a un petit souci avec ce qu'on a fait, lié aux limitations du [calcul en virgule
flottante](https://fr.wikipedia.org/wiki/Virgule_flottante), qui est la façon dont nos machines
représentent et manipulent les nombres non-entiers. Imaginez qu'on essaie de calculer la
vraisemblance d'une phrase de $100$ mots, ou la probabilité de chaque nouveau mot est de $0.0002$
(ou `2e-4`, soit $2×10^{-4}$)

```python
accumulator = 1.0
for i in range(100):
    accumulator *= 0.0002
    print(accumulator)
```

Vous voyez le problème ? On a multiplié que des nombres non-nuls, le résultat ne devrait donc pas
pouvoir être $0$. Et pourtant, comme la précision de la machin est limitée, c'est bien ce qu'on
finit par obtenir. (On parle d'*underflow*).


On va utiliser une astuce de maths ici : on va passer au
[logarithme](https://fr.wikipedia.org/wiki/Logarithme). C'est-à-dire qu'au lieu de calculer la
vraisemblance $p_s$ de la phrase $s$, on va calculer sa **log-vraisemblance** $\log(p_s)$.


Pour l'instant, pas besoin de se casser la tête sur ce que c'est que le logarithme. Les points
importants ici sont :

- Il existe une fonction (au sens mathématique), notée $\log$, elle est accessible en Python comme
  `math.log`.
- $\log$ est strictement croissante, c'est-à-dire que si $x < y$, alors $\log(x) < \log(y)$. Ça nous
  assure que si on trouve que si un modèle est meilleur qu'un autre en termes de log-vraisemblance,
  il l'est aussi en termes de vraisemblance tout cours.
- $\log(a×b) = \log(a) + \log(b)$, ce qui va nous permettre de calculer facilement $\log(p_s)$ et
  diminuant très grandement le risque d'*underflow*.

```python
import math

print(math.log(2))
print(math.log(1))
```

Attention en revanche : $\log(0) = -∞$, donc ceci ne marche pas

```python
print(math.log(0))
```

Mais c'est pas grave : on a par construction pas de mots de probabilité $0$.

### 🤘🏻 Calculer la perplexité 🤘🏻

1\. Écrire une fonction `sent_loglikelihood`, qui les mêmes arguments que `sent_likelihood` et
renvoie la **log-vraisemblance** de cette phrase.


```python
def sent_loglikelihood(ngram_probs, sent, n):
    pass  # À toi de coder

assert sent_loglikelihood(trigram_probs, ["pénitentes", ",", "que", "prenez-vous", "?"], 3) == -30.86945552941164
```


2\. Écrire une fonction `avg_log_likelihood`, qui prend qui prend en argument des probas de
n-grammes, un chemin vers un corpus et `n` et renvoie la vraisemblance moyenne de ce corpus estimée
par le modèle à n-grammes correspondant à ces probas (autrement dit, la moyenne des
log-vraisemblances de phrases). Testez sur *Le Ventre de Paris* (non, on l'a dit, c'est pas une
évaluation juste, mais ça va nous permettre de voir si ça marche).

```python
def avg_log_likelihood(ngram_probs, file_path, n):
    pass  # À toi de coder

assert avg_log_likelihood(bigram_probs, "data/zola_ventre-de-paris.txt", 2) == -56.321217776181875
assert avg_log_likelihood(trigram_probs, "data/zola_ventre-de-paris.txt", 3) == -81.20968449380536
assert avg_log_likelihood(pentagram_probs, "data/zola_ventre-de-paris.txt", 5) == -88.25016939038316
```


## Mots inconnus et évaluation en général

À vous de jouer maintenant !

Coder l'évaluation sur [*Le Rouge et le Noir*](data/rouge_noir.txt) (il
se trouve dans `"data/rouge_noir.txt"`) des modèles de langues appris sur *Le Ventre de Paris*.
Attention, vous allez vous heurter à des problèmes de vocabulaires incompatibles et de n-grammes
inexistants. Pour les résoudre, vous allez devoir vous servir des infos des sections 3.4 et 3.5.1 de
[*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/3.pdf).
