---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cours-ml
    language: python
    name: python3
---

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
TP 4 : Régression logistique
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

```python
from IPython.display import display, Markdown
```

```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import tol_colors as tc
```

## 🧠 Exo 🧠

### 1. Vectoriser un document

Regardez la tête du lexique [VADER](https://github.com/cjhutto/vaderSentiment) (vous le
trouverez aussi dans [`data/vader_lexicon.txt`](data/vader_lexicon.txt)). Il contient une liste de
mots (première colonne) avec pour chaque mot $w$ une polarité (deuxième colonne) $p(w)$, positive ou
négative.

1\. Écrivez une fonction qui lit le lexique VADER et renvoie un `dict` associant à chaque mot sa
polarité.

```python
def read_vader(vader_path: str) -> dict[str, float]:
    pass  # À vous de jouer
```

(Il y a des annotations de types dans la cellule précédente, je vais essayer d'en mettre autant que
possible dans les TPSs futurs. Il est vivement conseillé de lire
<https://realpython.com/lessons/type-hinting/> et <https://realpython.com/python-type-checking/> et
de parcourir [la doc](https://docs.python.org/3/library/typing.html) à ce sujet).

Étant donné un document $d$, vu comme un sac de mots de longueur $N$, on peut utiliser VADER pour
donner une représentation très (trop) simple de $d$ : un vecteur de deux *features* : sa polarité
positive moyenne $g$ et sa polarité négative moyenne $b$, définies par :

$$
\begin{equation}
    \begin{aligned}
        g &= \frac{1}{N}\sum_{w∈d} \max(p(w), 0)\\
        b &= \frac{1}{N}\sum_{w∈d} (-\min(p(w), 0))
    \end{aligned}
\end{equation}
$$

(notez bien que $g$ et $b$ sont tous les deux des nombres **positifs**)


2\. Écrire une fonction qui prend en entrée un texte en anglais et le dictionnaire précédent et
renvoie cette représentation. On pourra supposer que les mots inconnus ont une polarité de $0$.

```python
def featurize(doc: str, lexicon: dict[str, float]) -> np.ndarray:
    pass # À vous de jouer !
```

Vous aurez besoin d'extraire les mots du texte, vous pouvez le faire avec nltk, spacy, ou comme
scikit-learn, sauvagement avec une regex comme `\b\w+\b'`.

```python
lexicon = read_vader("data/vader_lexicon.txt")
doc = "I came in in the middle of this film so I had no idea about any credits or even its title till I looked it up here, where I see that it has received a mixed reception by your commentators. I'm on the positive side regarding this film but one thing really caught my attention as I watched: the beautiful and sensitive score written in a Coplandesque Americana style. My surprise was great when I discovered the score to have been written by none other than John Williams himself. True he has written sensitive and poignant scores such as Schindler's List but one usually associates his name with such bombasticities as Star Wars. But in my opinion what Williams has written for this movie surpasses anything I've ever heard of his for tenderness, sensitivity and beauty, fully in keeping with the tender and lovely plot of the movie. And another recent score of his, for Catch Me if You Can, shows still more wit and sophistication. As to Stanley and Iris, I like education movies like How Green was my Valley and Konrack, that one with John Voigt and his young African American charges in South Carolina, and Danny deVito's Renaissance Man, etc. They tell a necessary story of intellectual and spiritual awakening, a story which can't be told often enough. This one is an excellent addition to that genre."
doc_features = featurize(doc, lexicon)
doc_features
```

### 2. Vectoriser un corpus

Utiliser la fonction précédente pour vectoriser [le mini-corpus IMDB](data/imdb_smol.tar.gz) :
générez un `DataFrame` polars où chaque document est représenté par une ligne, les colonnes étant
`g`, `b` et `cls`, la classe (`"pos"` ou `"neg"`) du document (qui vous est donnée par son
sous-dossier).

Rappel : vous pouvez bien sûr commencer par vectoriser chaque document et ensuite construire un
DataFrame, mais en ce qui me concerne, je préfère faire l'inverse : construire d'abord un DataFrame
avec les contenus des documents et leurs classes et construire les colonnes `g` et `b` en utilisant
[`map_elements()`](https://docs.pola.rs/user-guide/expressions/user-defined-functions/).

(Réécrire `featurize` comme une expression polars, c'est *possible*, c'est pas très agréable, mais
c'est plus rapide. À vous de voir selon vos goûts.)

```python
def featurize_dir(corpus_root_path: str, lexicon: dict[str, float]) -> pl.DataFrame:
    pass # À vous!

# On réutilise le lexique précédent
featurize_dir("data/imdb_smol", lexicon)
```

## Visualisation

Comment se répartissent les documents du corpus avec la représentation qu'on a choisi ?

```python
lexicon = read_vader("data/vader_lexicon.txt")
imdb_features = featurize_dir("data/imdb_smol", lexicon)

X = imdb_features.get_column("g").to_numpy()
Y = imdb_features.get_column("b").to_numpy()
H = imdb_features.get_column("cls").to_numpy()

fig = plt.figure(dpi=200)
sns.scatterplot(x=X, y=Y, hue=H, s=5)
plt.show()
```

On voit des tendances qui se dégagent, mais clairement ça va être un peu coton

## Classifieur linéaire

On considère des vecteurs de *features* de dimension $n$

$$\mathbf{x} = (x₁, …, x_n)$$

Un vecteur de poids de dimension $n$

$$\mathbf{α} = (α₁, …, α_n)$$

et un biais $β$ scalaire (un nombre quoi).

Pour réaliser une classification on considère le nombre $z$

$$z=α₁×x₁ + … + α_n×x_n + β = \sum_iα_ix_i + β$$

Ce qu'on note aussi

$$z = \mathbf{α}⋅\mathbf{x}+β$$

$\mathbf{α}⋅\mathbf{x}$ se lit « alpha scalaire x », on parle de *produit scalaire* en français et de
*inner product* en anglais.

(ou pour les mathématicien⋅nes acharné⋅es $z = \langle α\ |\ x \rangle + β$)

Quelle que soit la façon dont on le note, on affectera à $\mathbf{x}$ la classe $0$ si $z < 0$ et la
classe $1$ sinon.

## 😴 Exo 😴

### 1. Une fonction affine

Écrire une fonction qui prend en entrée un vecteur de features et un vecteur de poids sous forme de
tableaux numpy $x$ et $α$ de dimensions `(n,)` et un biais $β$ sous forme d'un tableau numpy de
dimensions `(1,)` et renvoie $z=\sum_iα_ix_i + β$.

```python
def affine_combination(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # Attention, j'ai typé la valeur de retour comme un ndarray, mais il devra être
    # de forme (1,)
    pass # À vous de jouer !

affine_combination(
    np.array([2, 0, 2, 1]),
    np.array([-0.2, 999.1, 0.5, 2]),
    np.array([1]),
)
```

### 2. Un classifieur linéaire

1\. Écrire un classifieur linéaire qui prend en entrée des vecteurs de features à deux dimensions
précédents et utilise les poids respectifs $0.6$ et $-0.4$ et un biais de $-0.01$.

```python
lexicon = read_vader("data/vader_lexicon.txt")
imdb_features = featurize_dir("data/imdb_smol", lexicon)
```

```python
def hardcoded_classifier(x: np.ndarray) -> str:
    return "pos"  # À vous de jouer

hardcoded_classifier(np.array([2.7, 1.3]))
```

2\. Appliquez ce classifieur sur le mini-corpus IMDB qu'on a vectorisé et calculez son exactitude.

```python
# Faites marcher cette cellule. Il faudra sans doute écrire une fonction.

classifier_accuracy(np.array([0.6, -0.4]), np.array(-0.01), imdb_features)
```

## Classifieur linéaire ?

Pourquoi linéaire ? Regardez la figure suivante qui colore les points $(x,y)$ du plan en fonction de
la valeur de $z$.

```python
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = (0.6*X - 0.4*Y) - 0.01

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

Ou encore plus clairement, si on représente la classe assignée

```python

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = (0.6*X - 0.4*Y) -0.01 > 0.0

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

On voit bien que la frontière de classification est une droite, *a line*. On a donc un *linear*
classifier : un classifieur linéaire (même si en français on dirait qu'il s'agit d'une fonction
*affine*).


Qu'est-ce que ça donne si on superpose avec notre corpus ?

```python
fig = plt.figure(dpi=200)

x = np.linspace(0, 0.4, 1000)
y = np.linspace(0, 0.4, 1000)
X, Y = np.meshgrid(x, y)
Z = (0.6*X - 0.4*Y) -0.01 > 0.0

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))

X = imdb_features.get_column("g").to_numpy()
Y = imdb_features.get_column("b").to_numpy()
H = imdb_features.get_column("cls").to_numpy()

plt.scatter(x=X, y=Y, c=H, cmap="viridis", s=5)

plt.show()
```

Pas si surprenant que nos résultats ne soient pas terribles…

## La fonction logistique


$$σ(z) = \frac{1}{1 + e^{−z}} = \frac{1}{1 + \exp(−z)}$$

Elle permet de *normaliser* $z$ : $z$ peut être n'importe quel nombre entre $-∞$ et $+∞$, mais on
aura toujours $0 < σ(z) < 1$, ce qui permet de l'interpréter facilement comme une *vraisemblance*.
Autrement dit, $σ(z)$ sera proche de $1$ s'il paraît vraisemblable que $x$ appartienne à la classe
$1$ et proche de $0$ sinon.

## 📈 Exo 📈


1\. Définir la fonction `logistic` qui prend en entrée un tableau numpy $z=[z_1, …, z_n]$ et
renvoie le tableau $[σ(z_1), … , σ(z_n)]$.

```python
def logistic(z: np.ndarray) -> np.ndarray:
    pass  # À vous
```

2\. Tracer avec matplotlib la courbe représentative de la fonction logistique.

```python

```

## Régression logistique

Formellement : on suppose qu'il existe une fonction $f$ qui prédit parfaitement les classes, donc
telle que pour tout couple exemple/classe $(x, y)$ avec $y$ valant $0$ ou $1$, $f(x) = y$. On
voudrait approcher cette fonction par une fonction $M$ de la forme

$$M(x) = σ(\langle α\ |\ x \rangle + β)$$

Si on choisit les poids $α$ et le biais $β$ tels que $M$ soit la plus proche possible de $f$ sur
notre ensemble d'apprentissage, on dit que $M$ est la *régression logistique de $f$* sur cet
ensemble.

Un classifieur logistique, c'est simplement un classifieur qui pour un exemple $x$ renvoie $0$ si
$M(x) < 0.5$ et $1$ sinon. Il a exactement les mêmes capacités de discrimination qu'un classifieur
linéaire (sa frontière de décision est la même et il ne sait donc pas prendre de décisions plus
complexes), mais on a accès à un score qui peut être interprété comme la confiance qu'il a dans sa
décision.


Par exemple voici la confiance que notre classifieur codé en dur a en ses décisions

```python
def classifier_confidence(x: np.ndarray) -> np.ndarray:
    return logistic(affine_combination(x, np.array([0.6, -0.4]), -0.01))

doc_features = featurize(doc, lexicon)
M_x = classifier_confidence(doc_features)
display(M_x)
display(Markdown(f"Le classifieur est sûr à {M_x:.06%} que ce document est dans la classe $1$."))
display(Markdown(f"Autrement dit, d'après le classifieur, la classe $1$ a {M_x:.06%} de vraisemblance pour ce document"))
```


Quelle est la vraisemblance de la classe $0$ (*review* négative) ? Et bien le reste

```python
1.0 - classifier_confidence(doc_features)
```

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
que jugée par le classifieur sera de $50.49\%$. Un classifieur parfait obtiendrait $100\%$, un
classifieur qui prendrait systématiquement la mauvaise décision $0\%$ et un classifieur aléatoire
uniforme $50\%$ (puisque notre corpus a autant d'exemples de chaque classe).


Moralité : nos poids ne sont pas très bien choisis, et notre préoccupation dans la suite va être de
chercher comment choisir des poids pour que la confiance moyenne de la classe correcte soit aussi
haute que possible.


**Attention** le score que nous donne le classifieur peut être vu sa confiance dans la décision prise **mais** ça ne veut pas dire que ce score a beaucoup de valeur pour étudier le modèle. De fait, la confiance en question n'est que très rarement corrélée avec l'exactitude des précisions. 

Autrement dit : **quand un classifieur logistique se trompe, il a tendance à le faire avec beaucoup de confiance mal placée.**

## Fonction de coût

On a dit que notre objectif était

> Chercher les poids $α$ et le biais $β$ tels que $M$ soit la plus proche possible de $f$ sur notre
ensemble d'apprentissage

On formalise « être le plus proche possible » de la section précédente comme **minimiser** une
certaine fonction de **coût** (*loss*) $\mathcal{L}$ qui mesure l'erreur faite par le classifieur.

On déifinit souvent $\mathcal{L}$ en évaluant d'abord un coût *local* $L$ pour chaque exemple :

$$L(M(x), y) = \text{l'écart entre la classe $ŷ=M(x)$ prédite par $M$ pour $x$ et la classe correcte $y$}$$

Puis, étant donné un ensemble de test $\mathcal{D}_t = \{(x₁, y₁), …, (x_n, y_n)\}$, on définit $\mathcal{L}$ comme le coût total :

$$\mathcal{L}(M, \mathcal{D}_t) = \sum_i L(M(xᵢ), yᵢ)$$

Plus $\mathcal{L}$ sera bas, meilleur sera notre classifieur.




Dans le cas de la régression logistique, on va s'inspirer de ce qu'on a vu dans la section
précédente et utiliser la *log-vraisemblance négative* (*negative log-likelihood*) :

On définit la *vraisemblance* $V$ comme précédemment par
$$
V(M(x), y) =
    \begin{cases}
        M(x) & \text{si $y = 1$}\\
        1-M(x) & \text{sinon}
    \end{cases}
$$

Intuitivement, il s'agit du score attribué par le modèle à la classe correcte $y$. Il ne
s'agit donc pas d'un coût, mais d'un *gain* (si sa valeur est haute, c'est que le modèle est bon)

La *log-vraisemblance négative* $L$ est alors définie par

$$L(M(x), y) = -\log(V(M(x), y))$$

Le $\log$ est là pour plusieurs raisons, calculatoires et théoriques et le $-$ à
s'assurer qu'on a bien un coût (plus la valeur est basse, meilleur le modèle est).

<details>
<summary>Details</summary>
Entre autres, parce qu'une somme de $\log$-vraisemblances peut
être vue comme le $\log$ de la probabilité d'une conjonction d'événements indépendants. Mais surtout
parce qu'il rend la fonction de coût **convexe** par rapport à $(α, β)$.

Une interprétation possible : $L(M(x), y)$, c'est la
[surprise](https://en.wikipedia.org/wiki/Information_content) de $y$ au sens de la théorie de
l'information. Autrement dit : si j'estime qu'il y a une probabilité $a$ d'observer la classe $y$,
$L(a, y)$ mesure à quel point il serait surprenant d'observer effectivement $y$.
</details>


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

On peut aussi vérifier facilement que $L(M(x), 1)$ est décroissant par rapport à $M(x)$ et que $L(1-M(x), 0)$
est croissant par rapport à $M(x)$. Autrement dit, plus le classifieur juge que la classe correcte est
vraisemblable plus le coût $L$ est bas.


Enfin, on peut écrire $L$ en une ligne : pour un exemple $x$, le coût de l'exemple $(x, y)$ est

$$L(M(x), y) = -\log\left[M(x)×y + (1-M(x))×(1-y)\right]$$

C'est une astuce : comme $y$ vaut soit $0$ soit $1$, on a  soit $y=0$, soit $1-y=0$, et donc la
somme dans le $\log$ se simplifie dans tous les cas.

Une dernière façon de l'écrire en une ligne :

$$L(M(x), y) = -\log\left[M(x)\mathbb{1}_{y=1} + (1-M(x))\mathbb{1}_{y=0}\right]$$

<details>
<summary>Si vous lisez SLP</summary>
La formule diffère un peu de celle de *Speech and Language Processing*, mais les résultats sont les
mêmes et celle-ci est mieux pour notre problème !

En fait la leur est la formule générale de l'entropie croisée pour des distributions de proba
à support dans $\{0, 1\}$, ce qui est une autre intuition pour cette fonction de coût, mais ici elle
nous complique la vie.
</details>

## 📉 Exo 📉

Écrire une fonction qui prend en entrée

- Un vecteur de features $x$ de taille $n$
- Un vecteur de poids $α$ de taille $n$ et un biais $β$ (de taille $1$)
- Une classe cible $y$ ($0$ ou $1$)

Et renvoie la log-vraisemblance négative du classifieur logistique de poids $(α, β)$ pour l'exemple
$(x, y)$.

Servez-vous en pour calculer le coût du classifieur de l'exercise précédent sur le mini-corpus IMDB.

Le résultat devrait ressembler à ça

```python
loss_on_imdb(np.array([0.6, -0.4]), -0.01, imdb_features)
```

<!-- #region -->
## Descente de gradient

### Principe général

L'**algorithme de descente de gradient** est la clé de voute de l'essentiel des travaux en
apprentissage artificiel moderne. Il s'agit d'un algorithme itératif qui étant donné un modèle
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
fig = plt.figure(figsize=(20, 20), dpi=200)
ax = plt.axes(projection='3d')

r = np.linspace(0, 8, 100)
p = np.linspace(0, 2*np.pi, 100)
R, P = np.meshgrid(r, p)
Z = -np.cos(R)/(1+0.5*R**2)

X, Y = R*np.cos(P), R*np.sin(P)

ax.plot_surface(X, Y, Z, cmap=tc.tol_cmap("sunset"), edgecolor="none", rstride=1, cstride=1)
# ax.plot_wireframe(X, Y, Z, color='black')

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
fonction $f$ dont le gradient est donné par `grad_f`, ça donne l'algo suivant

```python
def descent(grad_f: np.ndarray, theta_0: np.ndarray, learning_rate: float, n_steps: int) -> np.ndarray:
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
- `n_steps` est le nombre d'étapes d'optimisations. Dans un problème d'apprentissage, c'est aussi le
  nombre de fois où on aura parcouru l'ensemble d'apprentissage et on parle souvent d'**epoch**

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
  $\frac{∂f(θ)}{∂θ_i}$, la **dérivée partielle** de $f(θ)$ par rapport à $θ_i$, est la $i$-ème
  coordonnées du gradient de $f$.
- **Le taux d'apprentissage** est souvent noté $η$ (ou $α$ mais ici c'est déjà quelque chose)

L'étape de mise à jour de $θ$ peut donc se noter

$$θ ← θ - η \operatorname{grad}f


### Descente de gradient stochastique

Rappelez-vous, on a dit que notre fonction de coût, c'était

$$\mathcal{L} = \sum_i L(M(xᵢ), yᵢ)$$

et on cherche la valeur du paramètre $θ = (α_1, …, α_n, β)$ tel que $\mathcal{L}$ soit le plus petit
possible.


On peut utilise la propriété d'**additivité** du gradient : pour deux fonctions $f$ et $g$, on a

$$\operatorname{grad}(f+g) = \operatorname{grad}f + \operatorname{grad}g$$

Donc ici

$$\operatorname{grad}_{θ}\mathcal{L} = \sum_i \operatorname{grad}_{θ}L(M(xᵢ), yᵢ)$$

<!-- #region -->
Si on dispose d'une fonction `grad_L` qui, étant donnés $M(x_i)$ et $y_i$, renvoie
$\operatorname{grad}_{θ}L(M(x_i), y_i)$, l'algorithme de descente du gradient devient alors

```python
def descent(train_set: np.ndarray, theta_0: np.ndarray, learning_rate: float, n_steps: int) -> np.ndarray:
    theta = theta_0
    for _ in range(n_steps):
        alpha = theta[:-1]
        beta = theta[-1]
        partial_grads = []
        for (x, y) in train_set:
            # On calcule g(x)
            M_x = logistic(np.inner(alpha, x)+beta)
            # On calcule le gradient de L(g(x), y))
            partial_grads.append(grad_L(M_x, y))
        # On trouve la direction de plus grande pente
        steepest_direction = -np.sum(partial_grads)
        # On fait quelques pas dans cette direction
        theta += learning_rate*steepest_direction
        
    return theta
```
<!-- #endregion -->

Pour chaque étape, on doit calculer tous les $M(x_i)$ et $\operatorname{grad}_{θ}L(M(x_i), y_i)$. C'est
très couteux, il doit y avoir moyen de faire mieux.


Si les $L(M(xᵢ), yᵢ)$ étaient indépendants, ce serait plus simple : on pourrait les optimiser
séparément.


Ce n'est évidemment pas le cas : si on change $M$ pour que $M(x_0)$ soit plus proche de $y_0$, ça
changera aussi la valeur de $M(x_1)$.


**Mais on va faire comme si**

<!-- #region -->
C'est une approximation sauvage, mais après tout on commence à avoir l'habitude. On va donc suivre
l'algo suivant

```python
def descent(train_set: np.ndarray, theta_0: np.ndarray, learning_rate: float, n_steps: int) -> np.ndarray:
    theta = theta_0
    for _ in range(n_steps):
        for (x, y) in train_set:
            alpha = theta[:-1]
            beta = theta[-1]
            # On calcule g(x)
            M_x = logistic(np.inner(alpha, x)+beta)
            # On trouve la direction de plus grande pente
            steepest_direction = -grad_L(M_x, y)
            # On fait quelques pas dans cette direction
            theta += learning_rate*steepest_direction

    return theta
```
<!-- #endregion -->

Faites bien attention à la différence : au lieu d'attendre d'avoir calculé tous les
$\operatorname{grad}L(M(x_i), y_i)$ avant de modifier $θ$, on va le modifier à chaque fois.


- **Avantage** : on modifie beaucoup plus souvent le paramètre, si tout se passe bien, on devrait
  arriver à une bonne approximation très vite.
- **Inconvénient** : il se pourrait qu'en essayant de faire baisser $L(M(x_0), y_0)$, on fasse
  augmenter $L(M(x_1), y_1)$.

Notre espoir ici, c'est que cette situation n'arrivera pas, et qu'un bon paramètre pour un certain
couple $(x, y)$ soit aussi un bon paramètre pour $tous$ les couples `(exemple, classe)`.


Ce nouvel algorithme s'appelle l'**algorithme de descente de gradient stochastique**, et il est
crucial pour nous, parce qu'on ne pourra en pratique quasiment jamais faire de descente de gradient
globale.


Il ne nous reste plus qu'à savoir comment on calcule `grad_L`. On ne fera pas la preuve, mais on a


$$
\begin{equation}
    \operatorname{grad}_θL(M(x), y) =
        \begin{pmatrix}
            \frac{∂L(M(x), y)}{∂α_1}\\
            \vdots\\
             \frac{∂L(M(x), y)}{∂α_n}\\
             \frac{∂L(M(x), y)}{∂β}\\
        \end{pmatrix}
\end{equation}
$$


Avec pour tout $i$

$$\frac{∂L(M(x), y)}{∂α_i} = (M(x)-y)x_i$$


et


$$\frac{∂L(M(x), y)}{∂β} = M(x)-y$$

<!-- #region -->
## 🧐 Exo 🧐

### 1. Calculer le gradient

Écrire une fonction qui calcule $\operatorname{grad}_θL(M(x), y)$.
<!-- #endregion -->

```python
def grad_L(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray, y: np.ndarray) -> np.ndarray:
    grad = np.zeros(alpha.size+beta.size)  # À vous !
    return grad

grad_L(np.array([5, 10]), np.array([0.6, -0.4]), np.array([-0.01]), 0)
```

### 2. Descendre le gradient

S'en servir pour apprendre les poids à donner aux *features* précédentes à l'aide du [mini-corpus
IMDB](data/imdb_smol.tar.gz) en utilisant l'algorithme de descente de gradient stochastique.

```python
def descent(
    featurized_corpus: pl.DataFrame,
    theta_0: np.ndarray,
    learning_rate: float,
    n_steps: int
) -> np.ndarray:
    theta = theta_0
    for _ in range(n_steps):
        pass  # À vous !


descent(imdb_features, np.array([0.6, -0.4, 0.0]), 0.001, 100)
```

## Régression multinomiale

Un dernier point : on a vu dans tout ceci comment utiliser la régression logistique pour un problème
de classification à deux classes. Comment on l'étend à $n$ classes ?


Réfléchissons déjà à quoi ressemblerait la sortie d'un tel classifieur :

Pour un problème à deux classes, le classifieur $M$ nous donne pour chaque exemple $x$ une
estimation $M(x)$ de la vraisemblance de la classe $1$, et on a vu que la vraisemblance de la classe
$0$ était nécessairement $1-g(x)$ pour que la somme des vraisemblances fasse 1.


On peut le présenter autrement : considérons le classifieur $K$ tel que pour tout exemple $x$

$$K(x) = \begin{pmatrix}1-M(x)\\M(x)\end{pmatrix}$$

$K$ nous donne un vecteur à deux coordonnées, $K_0(x)$ et $K_1(x)$, qui sont respectivement les
vraisemblances des classes $0$ et $1$.


Pour un problème à $n$ classes, on va vouloir une vraisemblance par classe, on va donc procéder de
la façon suivante :

On considère des poids $(α_1, β_1), …, (α_n, β_n)$. Ils définissent un classifieur linéaire.

En effet, si on considère les $z_i$ définis pour tout exemple $x$ par

$$
    \begin{cases}
        z_1 = α_1⋅x + β_1\\
        \vdots\\
        z_n = α_n⋅x + β_n
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


Pourquoi elle s'appelle *softmax* ? Considérez le vecteur $v = (0.1, -0.5, 2.1, 2, 1.6)$. Son
maximum $\max(v)$, c'est $2.1$, et ce qu'on appelle $\operatorname{argmax}(v)$, la position du
maximum, c'est $3$.

Pour *argmax* à la place d'une position, on peut aussi le voir comme un masque : $(0, 0, 1, 0, 0)$,
autrement dit un vecteur dont les valeurs sont $0$ pour chaque position, sauf la position du
maximum, qui contient un $1$ (on parle de représentation *one-hot*). Visualisons ces vecteur :

```python
v = np.array([0.1, -0.5, 2.1, 1.8, 0.6])
```

```python
plt.bar(np.arange(v.shape[0]), v)
plt.title("Coordonnées de $v$")
plt.show()
```


```python
plt.bar(np.arange(v.shape[0]), v == v.max())
plt.title("Coordonnées de $\operatorname{argmax}(v)$")
plt.show()
```

Et softmax ? Regardez (on l'importe depuis
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html), un des
adelphes de NumPy, pour ne pas avoir à le recoder nous-même).

```python
from scipy.special import softmax
plt.bar(np.arange(v.shape[0]), softmax(v))
plt.title("Coordonnées de $\operatorname{softmax}(v)$")
plt.show()
```

On voit que c'est un genre d'entre-deux entre $v$ et $\operatorname{argmax}(v)$ :

- La distribution des coordonnées ressemble à celle de $v$…
- mais les coordonnées sont toutes entre $0$ et $1$.
- Le max est tiré vers $1$ et toutes les autres coordonnées vers $0$.
- La somme des coordonnées fait $1$.

Si, si, je vous assure :

```python
softmax(v).sum()
```

À l'erreur d'arrondi en virgule flottante près on est bien.


Autrement dit, on a, comme pour la fonction logistique, une fonction qui *normalise* les valeurs
tout en préservant certaines propriétés.


Revenons à nos moutons : on définit enfin le classifieur logistique multinomial $f$ de la façon
suivante : pour tout exemple $x$, on a

$$f(x) = \operatorname{softmax}(α_1⋅x+β_1, …, α_n⋅x+β_n) = \left(\frac{e^{α_1⋅x+β_1}}{\sum_i
e^{α_i⋅x+β_i}}, …, \frac{e^{α_n⋅x+β_n}}{\sum_i e^{α_i⋅x+β_i}}\right)$$

et on choisit pour $x$ la classe

$$y = \operatorname{argmax}\limits_i f_i(x) = \operatorname{argmax}\limits_i
\frac{e^{α_i⋅x+β_i}}{\sum_j e^{α_j⋅x+β_j}}$$

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
            &= \frac{e^{α_1⋅x+β_1}}{e^{α_0⋅x+β_0}+e^{α_1⋅x+β_1}}\\
            &= \frac{1}{
                \frac{e^{α_0⋅x+β_0}}{e^{α_1⋅x+β_1}} + 1
            }\\
            &= \frac{1}{e^{(α_0⋅x+β_0)-(α_1⋅x+β_1)} + 1}\\
            &= \frac{1}{1 + e^{(α_0-α_1)⋅x+(β_0-β_1)}}\\
            &= σ((α_0-α_1)⋅x+(β_0-β_1))
    \end{align}
$$


Autrement dit, appliquer ce qu'on vient de voir pour le cas multinomial, si $n=2$, c'est comme
appliquer ce qu'on a vu pour deux classes, avec $w=α_0-α_1$ et $b=β_0-β_1$.

## La suite

Vous devriez maintenant avoir quelques idées de plusieurs concepts importants :

- Le concept de classifieur linéaire
- Le concept de fonction de coût
- L'algorithme de descente de gradient stochastique
- La fonction softmax

Pour la suite de vos aventures au pays des classifieurs logistiques, je vous recommande plutôt
d'utiliser [leur implémentation dans
scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
