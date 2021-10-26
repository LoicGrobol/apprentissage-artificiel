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

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
Cours 10 : Régression logistique
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-27
<!-- #endregion -->

```python
from IPython.display import display
```

## Classifieur linéaire

On considère des vecteurs de *features* de dimension $n$

$$\mathbf{x} = (x₁, …, x_n)$$

Un vecteur de poids de dimension $n$

$$\mathbf{w} = (w₁, …, w_n)$$

et un biais $b$ scalaire (un nombre quoi).

Pour réaliser une classification on considère le nombre $z$ (on parle parfois de *logit*)

$$z=w₁×x₁ + … + w_n×x_n + b = \sum_iw_ix_i + b$$

Ce qu'on note aussi

$$z = \mathbf{w}⋅\mathbf{x}+b$$

$\mathbf{w}⋅\mathbf{x}$ se lit « w scalaire x », on parle de *produit scalaire* en français et de *inner product* en anglais.

(ou pour les mathématicien⋅ne⋅s acharné⋅e⋅s $z = \langle w\ |\ x \rangle + b$)

Quelle que soit la façon dont on le note, on affectera à $\mathbf{x}$ la classe $0$ si $z < 0$ et la classe $1$ sinon.

## 😴 Exo 😴

Écrire une fonction qui prend en entrée un vecteur de features et un vecteur de poids sous forme de
tableaux numpy $x$ et $w$ de dimensions `(n,)` et un biais $b$ sous forme d'un tableau numpy de
dimensions `(1,)` et renvoie $z=\sum_iw_ix_i + b$.

```python
import numpy as np
```

```python
def affine_combination(x, w, b):
    pass # À vous de jouer !

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

```python
def affine_combination(x, w, b):
    res = np.zeros(1)
    for wi, xi in zip(w, x):
        res += wi*xi
    res += b
    return res

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

```python
def affine_combination(x, w, b):
    return np.inner(w, x) + b

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

## 🧠 Exo 🧠

1\. À l'aide d'un lexique de sentiment (par exemple
[VADER](https://github.com/cjhutto/vaderSentiment)), écrivez une fonction qui prend en entrée un
texte en anglais et renvoie sa représentation sous forme d'un vecteur de features à deux traits :
nombre de mots positifs et nombre de mot négatifs.

```python
import re

def poor_mans_tokenizer_and_normalizer(s):
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]

def read_vader(vader_path):
    res = dict()
    with open(vader_path) as in_stream:
        for row in in_stream:
            word, polarity, *_ = row.lstrip().split("\t", maxsplit=2)
            is_positive = float(polarity) > 0
            res[word] = is_positive
    return res

def featurize(text, lexicon):
    words = poor_mans_tokenizer_and_normalizer(text)
    features = np.empty(2)
    features[0] = sum(1 for w in words if lexicon.get(w, False))
    features[1] = sum(1 for w in words if not lexicon.get(w, True))
    return features

lexicon = read_vader("../../data/vader_lexicon.txt")
doc = "I came in in the middle of this film so I had no idea about any credits or even its title till I looked it up here, where I see that it has received a mixed reception by your commentators. I'm on the positive side regarding this film but one thing really caught my attention as I watched: the beautiful and sensitive score written in a Coplandesque Americana style. My surprise was great when I discovered the score to have been written by none other than John Williams himself. True he has written sensitive and poignant scores such as Schindler's List but one usually associates his name with such bombasticities as Star Wars. But in my opinion what Williams has written for this movie surpasses anything I've ever heard of his for tenderness, sensitivity and beauty, fully in keeping with the tender and lovely plot of the movie. And another recent score of his, for Catch Me if You Can, shows still more wit and sophistication. As to Stanley and Iris, I like education movies like How Green was my Valley and Konrack, that one with John Voigt and his young African American charges in South Carolina, and Danny deVito's Renaissance Man, etc. They tell a necessary story of intellectual and spiritual awakening, a story which can't be told often enough. This one is an excellent addition to that genre."
doc_features = featurize(doc, lexicon)
doc_features
```

2\. Appliquer la fonction précédente sur [le mini-corpus IMDB](../../data/imdb_smol.tar.gz)


Commençons par l'extraire

```bash
cd ../../local
tar -xzf ../data/imdb_smol.tar.gz 
ls -lah imdb_smol
```

Maintenant on parcourt le dossier pour construire nos représentations

```python
from collections import defaultdict
import pathlib  # Manipuler des chemins et des fichiers agréablement

def featurize_dir(corpus_root, lexicon):
    corpus_root = pathlib.Path(corpus_root)
    res = defaultdict(list)
    for clss in corpus_root.iterdir():
        # On peut aussi utiliser une compréhension de liste et avoir un dict pas default
        for doc in clss.iterdir():
            # `stem` et `read_text` c'est de la magie de `pathlib`, check it out
            res[clss.stem].append(featurize(doc.read_text(), lexicon))
    return res

# On réutilise le lexique précédent
imdb_features = featurize_dir("../../local/imdb_smol", lexicon)
imdb_features
```

3\. Écrire un classifieur logistique qui prend en entrée les vecteurs de features
précédents et utilise les poids respectifs $0.6$ et $-0.4$ et un biais de $0$. Appliquez ce
classifieur sur le mini-corpus IMDB et calculez son exactitude.


On commence par définir le classifieur : on va renvoyer `True` pour la classe positive et `False`
pour la classe négative.


```python
def hardcoded_classifier(x):
    return (
        np.inner(
            np.array([0.6, -0.4]),
            x,
        )
        > 0.0
    )


hardcoded_classifier(doc_features)
```

Maintenant on le teste

```python
correct_pos = sum(1 for doc in imdb_features["pos"] if hardcoded_classifier(doc))
print(f"Recall for 'pos': {correct_pos}/{len(imdb_features['pos'])}={correct_pos/len(imdb_features['pos']):.02%}")
correct_neg = sum(1 for doc in imdb_features["neg"] if not hardcoded_classifier(doc))
print(f"Recall for 'neg': {correct_neg}/{len(imdb_features['neg'])}={correct_neg/len(imdb_features['neg']):.02%}")
print(f"Accuracy: {correct_pos+correct_neg}/{len(imdb_features['pos'])+len(imdb_features['neg'])}={(correct_pos+correct_neg)/(len(imdb_features['pos'])+len(imdb_features['neg'])):.02%}")
```


## La fonction logistique


$$σ(z) = \frac{1}{1 + e^{−z}} = \frac{1}{1 + \exp(−z)}$$

Elle permet de *normaliser* $z$ : $z$ peut être n'importe quel nombre entre $-∞$ et $+∞$, mais on
aura toujours $0 < σ(z) < 1$, ce qui permet de l'interpréter facilement comme une *vraisemblance*.
Autrement dit, $σ(z)$ sera proche de $1$ s'il paraît vraisemblable que $x$ appartienne à la classe
$1$ et proche de $0$ sinon.

## 📈 Exo 📈

Tracer avec matplotlib la courbe représentative de la fonction logistique.

```python
def logistic(z):
    return 1/(1+np.exp(-z))
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 5000)
y = logistic(x)
plt.plot(x, y)
plt.xlabel("$x$")
plt.ylabel("$σ(x)$")
plt.title("Courbe représentative de la fonction logistique sur $[-10, 10]$")
plt.show()
```

## Régression logistique

Formellement : on suppose qu'il existe une fonction $f$ qui prédit parfaitement les classes, donc
telle que pour tout couple exemple/étiquette $(x, y)$ avec $y$ valant $0$ ou $1$, $f(x) = y$. On
approcher cette fonction par une fonction $g$ de la forme

$$g(x) = σ(w⋅x+b)$$

Si on choisit les poids $w$ et le biais $b$ tels que $g$ soit la plus proche possible de $f$ sur
notre ensemble d'apprentissage, on dit que $g$ est la *régression logistique de $f$* sur cet
ensemble.

Un classifieur logistique, c'est simplement un classifieur qui pour un exemple $x$ renvoie $0$ si
$g(x) < 0.5$ et $1$ sinon. Il a exactement les mêmes capacités de discrimination qu'un classifieur
linéaire (il ne sait pas prendre de décisions plus complexes), mais on peut interpréter la confiance
qu'il a dans sa décision.


Par exemple voici la confiance que notre classifieur codé en dur a en ses décisions

```python
def classifier_confidence(x):
    return logistic(
        np.inner(
            np.array([0.6, -0.4]),
            x,
        )
    )


classifier_confidence(doc_features)
```


Autrement dit, d'après notre classifieur, il y a un peu plus de $99.86\%$ de chances que le document
soit de la classe $1$ (review positive dans ce cas), ou encore, la vraisemblance de la classe $1$
est vraisemblable à plus de $99.68\%$.


Quelle est la vraisemblance de la classe $0$ (review négative) ? Et bien le reste

```python
1.0 - classifier_confidence(doc_features)
```

Soit un peu plus de $0.1%$.


Comme l'exemple en question appartient bien à cette classe, ça signifie que notre classifieur et
plutôt bon **sur cet exemple**. L'est-il sur le reste du corpus ?

```python
pos_confidence = sum(classifier_confidence(doc) for doc in imdb_features["pos"])
print(f"Average confidence for 'pos': {pos_confidence/len(imdb_features['pos']):.02%}")
neg_confidence = sum(1-classifier_confidence(doc) for doc in imdb_features["neg"])
print(f"Average confidence for 'neg': {neg_confidence/len(imdb_features['neg']):.02%}")
print(f"Average confidence for the correct class: {(pos_confidence+neg_confidence)/(len(imdb_features['pos']) + len(imdb_features['neg'])):.02%}")
```

Autrement dit, pour un exemple pris au hasard dans le corpus, la vraisemblance de sa classe telle
que jugée par le classifieur sera de $59.47\%$. Un classifieur parfait obtiendrait $100\%$, un
classifieur qui prendrait systématiquement la mauvaise décision $0\%$ et un classifieur aléatoire
uniforme $50\%$ (puisque notre corpus a autant d'exemples de chaque classe).


Moralité nos poids ne sont pas très bien choisis, et notre préoccupation dans la suite va être de
chercher comment choisir des poids pour que la confiance moyenne de la classe correcte soit aussi
haute que possible.

## Fonction de coût

On a dit que notre objectif était

> Chercher les poids $w$ et le biais $b$ tels que $g$ soit la plus proche possible de $f$ sur
notre ensemble d'apprentissage

On formalise « être le plus proche possible » de la section précédente comme minimiser une certaine
fonction de coût (*loss*) $L$ qui mesure l'erreur faite par le classifieur sur un exemple.

$$L(g(x), y) = \text{l'écart entre la classe prédite par $g$ pour $x$ et la classe correcte $y$}$$

Étant donné un ensemble de test $(x₁, y₁), …, (x_n, y_n)$, on estime l'erreur faite par le
classifieur logistique $g$ pour chaque exemple $(x_i, y_i)$ comme le coût local $L(g(xᵢ), yᵢ)$ et
son erreur sur tout l'ensemble de test par le coût global $\mathcal{L}$ :

$$\mathcal{L} = \sum_i L(g(xᵢ), yᵢ)$$

Plus $\mathcal{L}$ sera bas, meilleur sera notre classifieur.

Dans le cas de la régression logistique, on va s'inspirer de ce qu'on a vu dans la section
précédente et utiliser la *log-vraisemblance négative* (*negative log-likelihood*) :

On définit la *vraisemblance* $V$ comme précédemment par
$$
V(a, y) =
    \begin{cases}
        a & \text{si $y = 1$}\\
        1-a & \text{sinon}
    \end{cases}
$$

Intuitivement, il s'agit de la vraisemblance affectée par le modèle à la classe correcte $y$. Il ne
s'agit donc pas d'un coût, mais d'un *gain* (si sa valeur est haute, c'est que le modèle est bon)

La *log-vraisemblance négative* $L$ est alors définie par

$$L(a, y) = -\log(V(a, y))$$

Le $\log$ est là pour plusieurs raisons, calculatoires et théoriques<sup>1</sup> et le $-$ à
s'assurer qu'on a bien un coût (plus la valeur est basse, meilleur le modèle est).

<small>1. Entre autres, comme pour *Naïve Bayes*, parce qu'une somme de $\log$-vraisemblance peut
être vue comme le $\log$ de la probabilité d'une conjonction d'événements indépendants. Mais surtout
parce qu'il rend la fonction de coût **convexe** par rapport à $w$</small>.

Une interprétation possible : $L(a, y)$, c'est la
[surprise](https://en.wikipedia.org/wiki/Information_content) de $y$ au sens de la théorie de
l'information. Autrement dit : si j'estime qu'il y a une probabilité $a$ d'observer la classe $y$,
$L(a, y)$ mesure à quel point il serait surprenant d'observer effectivement $y$.


On peut vérifier qu'il s'agit bien d'un coût :

- C'est un nombre positif
- Si le classifieur prend une décision correcte avec une confiance parfaite le coût est nul :

  $$
    \begin{cases}
        L(1.0, 1) = -\log(1.0) = 0\\
        L(0.0, 0) = -\log(1.0-0.0) = -\log(1.0) = 0
    \end{cases}
  $$
- Si le classifieur prend une décision erronée avec une confiance parfaite le coût est infini :

  $$
    \begin{cases}
        L(0.0, 1) = -\log(0.0) = +\infty\\
        L(1.0, 0) = -\log(1.0-1.0) = \log(0.0) = +\infty
    \end{cases}
  $$

On peut aussi vérifier facilement que $L(a, 1)$ est décroissant par rapport à $a$ et que $L(1-a, 0)$
est croissant par rapport à $a$. Autrement dit, plus le classifieur juge que la classe correcte est
vraisemblable plus le coût $L$ est bas.


Enfin, on peut l'écrire $L$ en une ligne : pour un exemple $x$, le coût de l'exemple $(x, y)$ est

$$L(g(x), y) = -\log\left[g(x)×y + (1-g(x))×(1-y)\right]$$

C'est un *trick*, l'astuce c'est que comme $y$ vaut soit $0$ soit $1$, soit $y=0$, soit $1-y=0$ et
donc la somme dans le $\log$ se simplifie dans tous les cas. Rien de transcendant là-dedans.

La formule diffère un peu de celle de *Speech and Language Processing*, mais les résultats sont les
mêmes et celle-ci est mieux pour notre problème !

<small>En fait la leur est la formule générale de l'entropie croisée pour des distributions de proba
à support dans $\{0, 1\}$, ce qui est une autre intuition pour cette fonction de coût, mais ici elle
nous complique la vie.</small>

Une dernière façon de l'écrire en une ligne :

$$L(g(x), y) = -\log\left[g(x)\mathbb{1}_{y=1} + (1-g(x))\mathbb{1}_{y=0}\right]$$

## 📉 Exo 📉

Écrire une fonction qui prend en entrée

- Un vecteur de features $x$ de taille $n$
- Un vecteur de poids $w$ de taille $n$ et un biais $b$ (de taille $1$)
- Une classe cible $y$ ($0$ ou $1$)

Et renvoie la log-vraisemblance négative du classifieur logistique de poids $(w, b)$ pour l'exemple
$(x, y)$.

Servez-vous en pour calculer le coût du classifieur de l'exercise précédent sur le mini-corpus IMDB.

<!-- #region -->
## Descente de gradient

### Principe général

L'**algorithme de descente de gradient** est la clé de voute de l'essentiel des travaux
en apprentissage artificiel moderne. Il s'agit d'un algorithme itératif qui étant donné un modèle
paramétrisé et une fonction de coût (avec des hypothèses de régularité assez faibles) permet de
trouver des valeurs des paramètres pour lesquelles la fonction de coût est minimal.

On ne va pas rentrer dans les détails de l'algorithme de descente de gradient stochastique, mais
juste essayer de se donner quelques idées.

L'intuition à avoir est la suivante : si vous êtes dans une vallée et que vous voulez trouver
rapidement le point le plus bas, une façon de faire est de chercher la direction vers laquelle la
pente descend le plus vite, de faire quelques pas dans cette direction puis de recommencer. On parle
aussi pour cette raison d'**algorithme de la plus forte pente**.

Clairement une condition pour que ça marche peu importe le point de départ, c'est que la vallée
n'ait qu'un seul point localement le plus bas. Par exemple ça marche avec une vallée comme celle-ci
<!-- #endregion -->

```python
%matplotlib inline
import tol_colors as tc
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(20, 20), dpi=200)
ax = plt.axes(projection='3d')

r = np.linspace(0, 8, 100)
p = np.linspace(0, 2*np.pi, 100)
R, P = np.meshgrid(r, p)
Z = R**2 - 1

X, Y = R*np.cos(P), R*np.sin(P)

ax.plot_surface(X, Y, Z, cmap=tc.tol_cmap("sunset"), edgecolor="none", rstride=1, cstride=1)
ax.plot_wireframe(X, Y, Z, color='black')

plt.show()
```

Mais pas pour celle-là

```python
%matplotlib inline
import tol_colors as tc
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(20, 20), dpi=200)
ax = plt.axes(projection='3d')

r = np.linspace(0, 8, 100)
p = np.linspace(0, 2*np.pi, 100)
R, P = np.meshgrid(r, p)
Z = -np.cos(R)/(1+0.5*R**2)

X, Y = R*np.cos(P), R*np.sin(P)

ax.plot_surface(X, Y, Z, cmap=tc.tol_cmap("sunset"), edgecolor="none", rstride=1, cstride=1)
#ax.plot_wireframe(X, Y, Z, color='black')

plt.show()
```

OK, mais comment on trouve la plus forte pente en pratique ? En une dimension il suffit de suivre
l'opposé du nombre dérivé : <https://uclaacm.github.io/gradient-descent-visualiser/#playground>


En plus de dimensions, c'est plus compliqué, mais on peut s'en sortir en suivant le *gradient* qui
est une généralisation du nombre dérivé : <https://jackmckew.dev/3d-gradient-descent-in-python.html>


Ce qui fait marcher la machine c'est que **le gradient indique la direction dans laquelle la
fonction croît le plus vite**. Et que l'opposé du gradient indique la direction dans laquelle la
fonction décroît le plus vite.

(localement)

<!-- #region -->
Concrètement si on veut trouver $\theta$ tel que $f(\theta)$ soit minimale pour une certaine
fonction $f$ dont le gradient est donné par `grad_f` ça donne l'algo suivant

```python
def descent(grad_f, theta_0, learning_rate, n_steps):
    theta = theta_0
    for _ in range(n_steps):
        # On trouve la direction de plus grande pente
        steepest_direction = -grad_f(theta)
        # On fait quelques pas dans cette direction
        theta += learning_rate*steepest_direction
    return theta
```

Les *hyperparamètres* sont

- `theta_0` est notre point de départ, notre première estimation d'où se trouvera le minimum, que
  l'algorithme va raffiner. Évidemment si on a déjà une idée de vers où on pourrait le trouver, ça
  ira plus vite. Si on a aucune idée, on peut le prendre aléatoire.
- `learning_rate` ou « taux d'apprentissage » : de combien on se déplace à chaque étape. Si on le
  prend grand on arrive vite vers la région du minimum, on mettra longtemps pour en trouver une
  approximation précise. Si on le prend petit, ça sera l'inverse.
- `n_steps` est le nombre d'étapes d'optimisations. Dans un problème d'apprentissage, c'est aussi
  le nombre de fois où on aura parcouru l'ensemble d'apprentissage et on parle souvent d'**epoch**

Ici on se donne un nombre fixe d'epochs, une autre possibilité serait de s'arrêter quand on ne bouge
plus trop, par exemple avec une condition comme

```python
if np.max(grad_f(theta)) < 0.00001:
    break
```

dans la boucle et éventuellement avec une boucle infinie `while True`.
<!-- #endregion -->

Point notation :

- **Le gradient de $f$** est souvent noté $\nabla f$ ou $\operatorname{grad}f$, voire $\vec\nabla f$
  ou $\overrightarrow{\operatorname{grad}} f$ (pour dire que c'est un vecteur)
- Si $θ=(θ_1, …, θ_n)$, autrement dit si $f$ est une fonction de $n$ variables, on note
  $\operatorname{grad}f = \left(\frac{∂f(θ)}{∂θ_1}, …, \frac{∂f(θ)}{∂θ_n}\right)$. Autrement dit
  $\frac{∂f(θ)}{∂θ_i}$, la **dérivée partielle** de $f(θ)$ par rapport à $θ_i$ est la $i$-ème
  coordonnées du gradient de $f$.
- **Le taux d'apprentissage** est souvent noté $α$ ou $η$


### Descente de gradient stochastique

Rappelez-vous, on a dit que notre fonction de coût, c'était

$$\mathcal{L} = \sum_i L(g(xᵢ), yᵢ)$$

et on cherche la valeur du paramètre $θ = (w_1, …, w_n, b)$ tel que $\mathcal{L}$ soit le plus petit
possible.


On peut utilise la propriété d'**additivité** du gradient : pour deux fonctions $f$ et $g$, on a

$$\operatorname{grad}(f+g) = \operatorname{grad}f + \operatorname{grad}g$$

Donc ici

$$\operatorname{grad}\mathcal{L} = \sum_i \operatorname{grad}L(g(xᵢ), yᵢ)$$

<!-- #region -->
Si on dispose d'une fonction `grad_L` qui, étant donnés $g(x_i)$ et $y_i$, renvoie
$\operatorname{grad}L(g(x_i), y_i)$, l'algorithme de descente du gradient devient alors

```python
def descent(train_set, theta_0, learning_rate, n_steps):
    theta = theta_0
    for _ in range(n_steps):
        w = theta[:-1]
        b = theta[-1]
        partial_grads = []
        for (x, y) in train_set:
            # On calcule g(x)
            g = sigmoid(np.inner(w*x+b))
            # On calcule le gradient de L(g(x), y))
            partial_grads.append(grad_L(g, y))
        # On trouve la direction de plus grande pente
        steepest_direction = -np.sum(partial_grads)
        # On fait quelques pas dans cette direction
        theta += learning_rate*steepest_direction
        
    return theta
```
<!-- #endregion -->

Pour chaque étape, on doit calculer tous les $g(x_i)$ et $\operatorname{grad}L(g(x_i), y_i)$. C'est
très couteux, il doit y avoir moyen de faire mieux.


Si les $L(g(xᵢ), yᵢ)$ étaient indépendants, ce serait plus simple : on pourrait les optimiser
séparément.


Ce n'est évidemment pas le cas : si on change $g$ pour que $g(x_0)$ soit plus proche de $y_0$, ça
changera aussi la valeur de $g(x_1)$.


**Mais on va faire comme si**

<!-- #region -->
C'est une approximation sauvage, mais après tout on commence à avoir l'habitude. On va donc suivre
l'algo suivant

```python
def descent(train_set, theta_0, learning_rate, n_steps):
    theta = theta_0
    for _ in range(n_steps):
        for (x, y) in train_set:
            w = theta[:-1]
            b = theta[-1]
            # On calcule g(x)
            g = sigmoid(np.inner(w*x+b))
            # On trouve la direction de plus grande pente
            steepest_direction = -grad_L(g, y)
            # On fait quelques pas dans cette direction
            theta += learning_rate*steepest_direction
        
    return theta
```
<!-- #endregion -->

Faites bien attention à la différence : au lieu d'attendre d'avoir calculé tous les
$\operatorname{grad}L(g(x_i), y_i)$ avant de modifier $θ$, on va le modifier à chaque fois.


- **Avantage** : on modifie beaucoup plus souvent le paramètre, si tout se passe bien, on devrait
  arriver à une bonne approximation très vite.
- **Inconvénient** : il se pourrait qu'en essayant de faire baisser $L(g(x_0), y_0)$, on fasse
  augmenter $L(g(x_1), y_1)$.

Notre espoir ici c'est que cette situation n'arrivera pas, et qu'on bon paramètre pour un certain
couple $(x, y)$ c'est un bon paramètres pour $tous$ les couples `(exemple, classe)`.


Ce nouvel algorithme s'appelle l'**algorithme de descente de gradient stochastique**, et il est
crucial pour nous, parce qu'on ne pourra en pratique quasiment jamais faire de descente de gradient
globale.


Il ne nous reste plus qu'à savoir comment on calcule `grad_L`. On ne fera pas la preuve, mais on a


$$\frac{∂L(g(x), y)}{∂w_i} = (g(x)-y)x_i$$


et


$$\frac{∂L(g(x), y)}{∂b} = g(x)-y$$


Autrement dit on mettra à jour $w$ en calculant

$$w ← w -η×\operatorname{d}_wL(g(x), y) = w - η×(g(x)-y)x$$


<small>$\operatorname{d}_wL(g(x), y) = \left(\frac{∂L(g(x), y)}{∂w_1}, …, \frac{∂L(g(x),
y)}{∂w_n}\right)$ est la *différentielle partielle* de $L(g(x), y)$ par rapport à $w$.</small>


Et $b$ en calculant

$$b ← b -η×\frac{∂L(g(x), y)}{∂b} = b - η×(g(x)-y)$$

## 🧐 Exo 🧐

1\. Reprendre la fonction qui calcule la fonction de coût, mais faire en sorte qu'elle renvoie également
le gradient par rapport à $w$ et la dérivée partielle par rapport à $b$ en $(x, y)$.


2\. S'en servir pour apprendre les poids à donner aux features précédentes à l'aide du  [mini-corpus
IMDB](../../data/imdb_smol.tar.gz) en utilisant l'algorithme de descente de gradient stochastique.

## Régression multinomiale

Un dernier point : on a vu dans tout ceci comment utiliser la régression logistique pour un problème
de classification à deux classes. Comment on l'étend à $n$ classes ?


Réflechissons déjà à quoi ressemblerait la sortie d'un tel classifieur :

Pour un problème à deux classes, le classifieur $g$ nous donne pour chaque exemple $x$ une
estimation $g(x)$ de la vraisemblance de la classe $1$, et on a vu que la vraisemblance de la classe
$0$ était nécessairement $1-g(x)$ pour que la somme des vraisemblances fasse 1.


On peut le présenter autrement : considérons le classifieur $f$ tel que pour tout exemple $x$

$$f(x) = (1-g(x), g(x))$$

$f$ nous donne un vecteur à deux coordonnées, $f_0(x)$ et $f_1(x)$, qui sont respectivement les
vraisemblances des classes $0$ et $1$.


Pour un problème à $n$ classes, on va vouloir une vraisemblance par classe, on va donc procéder de
la façon suivante :

On considère des poids $(w_1, b_1), …, (w_n, b_n)$. Ils définissent un classifieur linéaire.

En effet, si on considère les $z_i$ définis pour tout exemple $x$ par

$$
    \begin{cases}
        z_1 = w_1⋅x + b_1\\
        \vdots\\
        z_n = w_n⋅x + b_1
    \end{cases}
$$

On peut choisir la classe $y$ à affecter à $x$ en prenant $y=\operatorname{argmax}\limits_i z_i$


Il reste à normaliser pour avoir des vraisemblances. Pour ça on utilise une fonction **très**
importante : la fonction $\operatorname{softmax}$, définie ainsi :

$$\operatorname{softmax}(z_1, …, z_n) = \left(\frac{e^{z_1}}{\sum_i e^{z_i}}, …,
\frac{e^{z_n}}{\sum_i e^{z_i}}\right)$$

Contrairement à la fonction logistique qui prenait un nombre en entrée et renvoyait un nombre,
$\operatorname{softmax}$ prend en entrée un **vecteur** non-normalisé et renvoie un vecteur
normalisé.


On définit enfin le classifieur logistique multinomial $f$ de la façon suivante : pour tout exemple
$x$, on a

$$f(x) = \operatorname{softmax}(w_1⋅x+b_1, …, w_n⋅x+b_n) = \left(\frac{e^{w_1⋅x+b_1}}{\sum_i
e^{w_i⋅x+b_i}}, …, \frac{e^{w_n⋅x+b_n}}{\sum_i e^{w_i⋅x+b_i}}\right)$$

et on choisit pour $x$ la classe

$$y = \operatorname{argmax}\limits_i f_i(x) = \operatorname{argmax}\limits_i
\frac{e^{w_i⋅x+b_i}}{\sum_j e^{w_j⋅x+b_j}}$$

Comme la fonction exponentielle est croissante, ce sera la même classe que le classifieur linéaire
précédent. Comme pour le cas à deux classe, la différence se fera lors de l'apprentissage. Je vous
laisse aller en lire les détails dans *Speech and Language Processing*, mais l'idée est la même : on
utilise la log-vraisemblance négative de la classe correcte comme fonction de coût, et on optimise
les paramètres avec l'algo de descente de gradient stochastique.


Un dernier détail ?

Qu'est-ce qui se passe si on prend ce qu'on vient de voir pour $n=2$ ? Est-ce qu'on retombe sur le
cas à deux classe vu précédemment ?


Oui, regarde : dans ce cas

$$
    \begin{align}
        f_1(x)
            &= \frac{e^{w_1⋅x+b_1}}{e^{w_0⋅x+b_0}+e^{w_1⋅x+b_1}}\\
            &= \frac{1}{
                \frac{e^{w_0⋅x+b_0}}{e^{w_1⋅x+b_1}} + 1
            }\\
            &= \frac{1}{e^{(w_0⋅x+b_0)-(w_1⋅x+b_1)} + 1}\\
            &= \frac{1}{1 + e^{(w_0-w_1)⋅x+(b_0-b_1)}}\\
            &= σ((w_0-w_1)⋅x+(b_0-b_1))
    \end{align}
$$


Autrement dit, appliquer ce qu'on vient de voir pour le cas multinomial, si $n=2$, c'est comme
appliquer ce qu'on a vu pour deux classes, avec $w=w_0-w_1$ et $b=b_0-b_1$.

## La suite

Vous êtes arrivé⋅e⋅s au bout de ce cours et vous devriez avoir quelques idées de plusieurs concepts
importants :

- Le concept de classifieur linéaire
- Le concept de fonction de coût
- L'algorithme de descente de gradient stochastique
- La fonction softmax

On reparlera de tout ça en temps utile. Pour la suite de vos aventures au pays des classifieurs
logistiques, je vous recommande plutôt d'utiliser [leur implémentation dans
scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
Maintenant que vous savez comment ça marche, vous pouvez le faire la tête haute. Bravo !

<small>Vous avez aussi découvert les premiers réseaux de neurones de ce cours et ce n'est pas
rien !</small>