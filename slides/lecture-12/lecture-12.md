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
Cours 12 : Réseaux de neurones
==============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-11-09
<!-- #endregion -->

```python
from IPython.display import display, Markdown
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Le perceptron simple

[![Schéma d'un neurone avec des légendes pour les organelles et les connexions importantes pour la
communication entre
neurones.](https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png)](https://commons.wikimedia.org/w/index.php?curid=28761830)

Un modèle de neurone biologique (plutôt sensoriel) : une unité qui reçoit plusieurs entrées $x_i$
scalaires (des nombres quoi), en calcule une somme pondérée $z$ (avec des poids $w_i$ prédéfinis) et
renvoie une sortie binaire $y$ ($1$ si $z$ est positif, $0$ sinon).


Autrement dit

$$\begin{align}
z &= \sum_i w_ix_i\\
y &=
    \begin{cases}
        1 & \text{si $z > 0$}\\
        0 & \text{sinon}
    \end{cases}
\end{align}$$

Formulé célèbrement par McCulloch et Pitts (1943) avec des notations différentes

**Attention** selon les auteurices, le cas $z=0$ est traité différemment, pour *Speech and Language Processing*, on renvoie $0$ dans ce cas, c'est donc la convention qu'on suivra, mais vérifiez à chaque fois.


On peut ajouter un terme de *biais* en fixant $x_0=1$ et $w_0=b$, ce qui donne

$$\begin{equation}
    z = \sum_{i=0}^n w_ix_i = \sum_{i=1}^n w_ix_i + b
\end{equation}$$

Ou schématiquement


![](figures/perceptron/perceptron.svg)


Ou avec du code

```python
def perceptron(inpt, weights):
    """Calcule la sortie du perceptron dont les poids sont `weights` pour l'entrée `inpt`
    
    Entrées :
    
    - `inpt` un tableau numpy de dimension $n$
    - `weights` un tableau numpy de dimention $n+1$
    
    Sortie: un tableau numpy de type booléen et de dimentions $0$
    """
    return (np.inner(weights[1:], inpt) + weights[0]) > 0
```

Implémenté comme une machine, le perceptron Mark I, par Rosenblatt (1958) :

[![Une photographie en noir et blanc d'une machine ressemblant à une grande armoire pleine de fils
électriques](https://upload.wikimedia.org/wikipedia/en/5/52/Mark_I_perceptron.jpeg)](https://en.wikipedia.org/wiki/File:Mark_I_perceptron.jpeg)


**Est-ce que ça vous rappelle quelque chose ?**


🤔


C'est un **classifieur linéaire** dont on a déjà parlé [précédemment](../lecture10/lecture10.md).


Les ambitions initiales étaient grandes

> *the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence.*  
> New York Times, rappporté par Olazaran (1996)


C'est par exemple assez facile de construire des neurones qui réalisent les opérations logiques élémentaires $\operatorname{ET}$, $\operatorname{OU}$ et $\operatorname{NON}$ :

```python
and_weights = np.array([-0.6, 0.5, 0.5])
print("x\ty\tx ET y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = perceptron([x_i, y_i], and_weights).astype(int)
        print(f"{x_i}\t{y_i}\t{out}")
```

<!-- TODO: ceci pourrait être un exo -->

```python
or_weights = np.array([-0.5, 1, 1])
print("x\ty\tx OU y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = perceptron([x_i, y_i], or_weights).astype(int)
        print(f"{x_i}\t{y_i}\t{out}")
```

```python
not_weights = np.array([1, -1])
print("x\tNON x")
for x_i in [0, 1]:
    out = perceptron([x_i], not_weights).astype(int)
    print(f"{x_i}\t{out}")
```

Mais on se heurte vite à des problèmes, même pour représenter les fonctions logiques les plus
basiques, comme $\operatorname{XOR}$ définie pour $x ∈ \{0, 1\}$ et  $y ∈ \{0, 1\}$ par :


$$\begin{equation}
    \operatorname{XOR}(x, y) = 
        \begin{cases}
            1 & \text{si $x ≠ y$}\\
            0 & \text{si $x = y$}
        \end{cases}
\end{equation}$$



Autrement dit, $\operatorname{XOR}(x, y)$ c'est vrai si $x$ est vrai ou si $y$ est vrai, mais pas si
les deux sont vrais en même temps.

```python
import tol_colors as tc

x = np.array([0, 1])
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)
Z = np.logical_xor(X, Y)

fig = plt.figure(dpi=200)

heatmap = plt.scatter(X, Y, c=Z, cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```


Si on l'étend à tout le plan pour mieux voir


```python
import tol_colors as tc

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = np.logical_xor(X > 0.5, Y > 0.5)

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

On voit clairement le hic : ce n'est pas un problème linéairement séparable, donc un classifieur linéaire ne sera jamais capable de le résoudre.


## Réseaux de neurones

Comment on peut s'en sortir ? En combinant des neurones !


On sait faire les portes logiques élémentaires $\operatorname{ET}$, $\operatorname{OU}$ et $\operatorname{NON}$, or on a

$$\begin{equation}
    x \operatorname{XOR} y = (x \operatorname{OU} y)\quad\operatorname{ET}\quad\operatorname{NON}(x \operatorname{ET} y)
\end{equation}$$

<small>
Ou en notation fonctionnelle

$$\begin{equation}
    \operatorname{XOR}(x, y) = \operatorname{ET}\left[\operatorname{OU}(x, y), \operatorname{NON}(\operatorname{ET}(x,y))\right]
\end{equation}$$
</small>


On peut donc avoir $\operatorname{XOR}$ non pas avec un seul neurone, mais avec plusieurs neurones mis en **réseau**

![](figures/xor/xor.svg)

Ou, en écrivant les termes de biais dans les neurones et en ajoutant un neurone pour servir de relai

![](figures/xor_ffnn/xor_ffnn.svg)

On voit ici appraître une structure en plusieurs couches (une d'entrée, une de sortie et trois intermédiaires) où chaque neurone prend en entrée les sorties de tous les neurones de la couche précédente.

On appelle cette structure un réseau de neurones **complètement connecté** ou **dense**. On parle aussi un peu abusivement de *perceptron multicouches*. En anglais _**multilayer perceptron**_ ou _**feedforward neural network**_.


Voyons sa frontière de décision

```python
import tol_colors as tc

def and_net(X, Y):
    return 0.5*X + 0.5*Y - 0.6 > 0

def or_net(X, Y):
    return X + Y - 0.5 > 0

def not_net(X):
    return -1*X + 1 > 0

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = and_net(or_net(X, Y), not_net(and_net(X, Y)))

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

Ça marche !


Enfin ça marche pour les coins, mais c'est tout ce qui nous intéressait !

## Les couches

Une autre façon de voir ces couches neuronales qui va être bien pratique pour la suite, c'est de voir chaque couche comme une fonction qui renvoie autant de sorties qu'elle a de neurones et prend autant d'entrées qu'il y a de neurones dans la couche précédente. Par exemple la première couche notre réseau $\operatorname{XOR}$ peut s'écrire comme :

```python
def layer1(inpt):
    output_1 = ((np.inner(inpt, np.array([0.5, 0.5])) + np.array(-0.6)) > 0).astype(int)
    output_2 = ((np.inner(inpt, np.array([1, 1])) + np.array(-0.5)) > 0).astype(int)
    return np.hstack([output_1, output_2])

display(layer1([1, 0]))
display(layer1([1, 1]))
```

La deuxième couche comme :

```python
def layer2(inpt):
    output_1 = ((np.inner(inpt, np.array([-1, 0])) + np.array(1)) > 0).astype(int)
    output_2 = ((np.inner(inpt, np.array([0, 1])) + np.array(0)) > 0).astype(int)
    return np.hstack([output_1, output_2])

display(layer2([0, 1]))
display(layer2([1, 1]))
```

Et la troisième couche comme

```python
def layer3(inpt):
    return ((np.inner(inpt, np.array([0.5, 0.5])) + np.array(-0.6)) > 0).astype(int)

display(layer3([1, 1]))
display(layer3([0, 1]))
```

Le réseau c'est donc

```python
def xor_ffnn(inpt):
    return layer3(layer2(layer1(inpt)))

print("x\ty\tx XOR y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = xor_ffnn([x_i, y_i]).astype(int)
        print(f"{x_i}\t{y_i}\t{out}")
```

Maintenant, si on regarde, ces fonctions ont toutes la même tête, on pourrait le faire en une seule

```python
def layer(inpt, weight1, bias1, weight2, bias2):
    output_1 = ((np.inner(inpt, weight1) + bias1) > 0).astype(int)
    output_2 = ((np.inner(inpt, weight2) + bias2) > 0).astype(int)
    return np.hstack([output_1, output_2])
                
layer([1, 0], [0.5, 0.5], -0.6, [1, 1], -0.5)
```

On peut rassembler ensemble les poids

```python
def layer(inpt, weight, bias):
    output_1 = ((np.inner(inpt, weight[0]) + bias[0]) > 0).astype(int)
    output_2 = ((np.inner(inpt, weight[1]) + bias[1]) > 0).astype(int)
    return np.hstack([output_1, output_2])
                
layer([0, 1], [[0.5, 0.5], [1, 1]], [-0.5, -0.6])
```

Et même, en l'écrivant comme des opérations matricielles

```python
def layer(inpt, weight, bias):
    output = ((np.matmul(weight, inpt) + bias) > 0).astype(int)
    return output
                
layer([0, 1], [[0.5, 0.5], [1, 1]], [-0.5, -0.6])
```

Cette dernière formulation est celle qu'on utilise en général, elle a le gros avantage de très bien se paralléliser, et même, si on dispose de matériel spécialisé (comme des cartes graphiques) de bénéficier d'accélérations supplémentaires (voir par exemple Vuduc et Choi ([2013](https://jeewhanchoi.github.io/publication/pdf/brief_history.pdf)) pour la culture).


Elle permet aussi de facilement manipuler les tailles des couches : une couche à $n$ entrées et $m$ sorties correspond à une matrice de poids de taille $m×n$ et un vecteur de biais de taille $m$.


Une dernière subtilité ? Pour matérialiser le concept de couche et éviter d'avoir à passer en permanence des poids, on utilise en général des classes. Voici notre réseau $\operatorname{XOR}$ réécrit en objet :

```python
class Layer:
    """Une couche neuronale complètement connectée"""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, inpt):
        return ((np.matmul(self.weight, inpt) + self.bias) > 0).astype(int)
    
layer1 = Layer(np.array([[0.5, 0.5], [1, 1]]), np.array([-0.6, -0.5]))
layer2 = Layer(np.array([[-1, 0], [0, 1]]), np.array([1, 0]))
layer3 = Layer(np.array([0.5, 0.5]), np.array(-0.6))

def xor_ffnn(inpt):
    return layer3(layer2(layer1(inpt)))

print("x\ty\tx XOR y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = xor_ffnn([x_i, y_i])
        print(f"{x_i}\t{y_i}\t{out}")
```

(admirez au passage l'usage de `__call__`)


On peut aussi imaginer un conteneur pour une réseau

```python
class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, inpt):
        res = inpt
        for l in self.layers:
            res = l(res)
        return res

xor_ffnn = Network([layer1, layer2, layer3])

print("x\ty\tx XOR y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = xor_ffnn([x_i, y_i])
        print(f"{x_i}\t{y_i}\t{out}")
```

Ça fait propre, non ?

## Non-linearités

Comme pour les classifieurs logistiques, on aime bien en général avoir une décision qui ne soit pas tout ou rien mais puisse prédire des nombres, pour ça on peut remplacer le `> 0` dans ce qui précède par la fonction logistique

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

class SigmoidLayer:
    """Une couche neuronale complètement connectée suivie de la fonction logistique"""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, inpt):
        return sigmoid(np.matmul(self.weight, inpt) + self.bias)
    
soft_and = SigmoidLayer(np.array([0.5, 0.5]), np.array(-0.6))

print("x\ty\tx soft_and y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = soft_and([x_i, y_i])
        print(f"{x_i}\t{y_i}\t{out}")
```

On peut aussi l'imaginer comme la succession d'une couche purement linéaire et d'une couche qui applique la fonction logistique sur ses entrées coordonnée par coordonnée :

```python
class LinearLayer:
    """Une couche neuronale linéaire complètement connectée"""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, inpt):
        return np.matmul(self.weight, inpt) + self.bias
    
class SigmoidLayer:
    """Une couche neuronale qui applique la fonction logistique aux coordonnées de son entrée"""
    # Pas besoin d'un `__init__` particulier ici, on a pas de paramètres à initialiser
    def __call__(self, inpt):
        return 1/(1+np.exp(-inpt))
    
soft_and = Network(
    [
        LinearLayer(np.array([0.5, 0.5]), np.array(-0.6)),
        SigmoidLayer(),
    ],
)

print("x\ty\tx soft_and y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = soft_and([x_i, y_i])
        print(f"{x_i}\t{y_i}\t{out}")
```

Dans le cas général, on dit que la fonction logistique dans ce réseau est une **non-linéarité** ou **activation**, c'est-à-dire une fonction non-linéaire appliquée coordonnée par coordonnée aux sorties d'une couche neuronale. On peut en choisir une autre, selon ce qu'on veut obtenir.

Pour les couches de sorties, c'est souvent l'application ciblée qui va conditionner ce choix, pour les couches internes, dites **couches cachées**, elle conditionnent la capacité d'apprentissage du réseau. Voici quelques uns des exemples les plus courants :

```python
x = np.linspace(-5, 5, 1000)

fig = plt.figure(dpi=200, constrained_layout=True)
axs = fig.subplots(3, 2)

axs[0, 0].plot(x, 1/(1+np.exp(-x)))
axs[0, 0].set_title("Fonction logistique")
axs[0, 1].plot(x, x > 0)
axs[0, 1].set_title("Fonction de Heavyside/échelon (unit step)")
axs[1, 0].plot(x, np.maximum(x, 0))
axs[1, 0].set_title("Rectifieur (ReLU)")
axs[1, 1].plot(x, np.tanh(x))
axs[1, 1].set_title("Tangente hyperbolique")
axs[1, 1].spines['left'].set_position('center')
axs[1, 1].spines['bottom'].set_position('zero')
axs[2, 0].plot(x, np.maximum(x, 0.1*x))
axs[2, 0].set_title("Leaky ReLU")
axs[2, 1].plot(x, np.maximum(x, 0.5*x*(1+np.tanh(np.sqrt(2*np.pi)*(x+0.044715*x**3)))))
axs[2, 1].set_title("GELU")

for ax in fig.get_axes():
    ax.spines["right"].set_color("none")
    ax.spines[bbb"top"].set_color("none")

plt.show()
```

Vous reconnaissez celle qu'on a utilisé dans notre réseau $\operatorname{XOR}$ ?


Et il y en a [plein](https://mlfromscratch.com/activation-functions-explained) d'autres.


En pratique, le choix de la bonne non-linéarité pour un réseau n'est pas encore bien compris : c'est un hyperparamètre à optimiser parmi d'autres. Ces dernières années on choisit plutôt par défaut la fonction rectifieur (dite un peu abusivement ReLU).


Mais au fait, pourquoi on s'embête avec ça ? Ça ne suffit pas des couches linéaires ?


Non.


Si on se limite à des couches linéaires, nos réseaux ne peuvent exprimer que des fonctions linéaires. Même si on peut faire beaucoup de choses avec, on a souvent besoin de plus. Voyez par exemple ce que ça donne dans [le bac à sable de Tensorflow](https://playground.tensorflow.org).


Si on utilise des non-linéarités, en revanche, nos réseaux deviennent beaucoup plus puissants. Beaucoup, **beaucoup** plus.


Le [**Théorème d'approximation universelle**](https://en.wikipedia.org/wiki/Universal_approximation_theorem), dont il existe de nombreuses versions (Pinkus ([1999](https://pinkus.net.technion.ac.il/files/2021/02/acta.pdf)) en fait une très bonne revue) dit en substance qu'à condition d'avoir assez de couches, ou des couches suffisament larges et d'utiliser des non-linéarités continues qui ne soient pas des polynômes, étant donnée une fonction continue $f$, on peut toujours trouver un réseau de neurones qui soit aussi près qu'on veut de $f$.


Bien que ça ne dise rien de la capacité des réseaux de neurones à *apprendre* des fonctions arbitraires, c'est une des motivations théoriques principales à leur utilisation : au moins, contrairement à un classifieur logistique par exemple, ils sont capables de représenter les fonctions qui nous intéressent.


Dernière précision : les couches linéaires et les non-linéarités par coordonnées ne sont pas les seules types de couches qu'on utilise en pratique. Notablement, pour construire des classifieurs multiclasses on utilise souvent la fonction $\operatorname{softmax}$ comme dernière couche.

$$\begin{equation}
    \operatorname{softmax}(z_1, …, z_n)
    = \left(
        \frac{e^{z_1}}{\sum_i e^{z_i}},
        …,
        \frac{e^{z_n}}{\sum_i e^{z_i}}
    \right)
\end{equation}$$


Au final, voici à quoi ressemble un classifieur neuronal classique.

```python
from scipy.special import softmax

class ReluLayer:
    """Une couche neuronale qui applique la fonction rectifieur"""
    def __call__(self, inpt):
        return np.maximum(inpt, 0)

class SoftmaxLayer:
    """Une couche neuronale qui applique la fonction softmax à son entrée"""
    def __call__(self, inpt):
        return softmax(inpt)
    
# Un réseau a une couche cachée de taille 32 qui prend en entrée des vecteurs
# de traits de dimension 16 et renvoie les vraisemblances de 8 classes.
# Les poids sont aléatoires
classifier = Network(
    [
        LinearLayer(np.random.normal(size=(32, 16)), np.random.normal(size=(32))),
        ReluLayer(),
        LinearLayer(np.random.normal(size=(8, 32)), np.random.normal(size=(8))),
        SoftmaxLayer(),
    ],
)

classifier([0, 2, 1, 3, 7, 0.1, 0.5, -12, 2, 1, -0.5, -1, 10
            , -2, 0.128, -8])
```

<small>[En pratique](https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html) comme $\operatorname{softmax}$ est toujours plutôt instable, on utilise plutôt $\log\operatorname{softmax}$, c'est toujours la même histoire.</small>

## Apprendre un réseau de neurones

Tout ça c'est bien gentil, mais encore une fois, on a choisi des poids à la main. Or notre objectif c'est d'**apprendre**.


Comment on apprend un réseau de neurone ? Comment on détermine les poids à partir de données ?


Et bien c'est toujours la même recette pour l'apprentissage supervisé :

- Déterminer une fonction de coût
- Apprendre par descente de gradient


Les fonctions de coût ressemblent très fort à celles d'autres techniques d'apprentissage. En TAL, comme on s'en sort toujours plus ou moins pour se rammener à de la classification, on va en général utiliser la $\log$-vraisemblance négative, comme pour les classifieurs logistiques.


Concrètement, qu'est-ce que ça donne ? Et bien si on a un réseau de neurones $f$ pour un problème à $n$ classes (donc qui renvoie en sortie des vecteurs normalisés de dimension $n$) et un exemple $(x, y)$ où $x$ est une entrée adaptée à $f$ et $1⩽y⩽n$ est la classe à prédire pour $x$, la loss de $f$ pour $(x, y)$ sera

$$\begin{equation}
    L(f, x, y) = -\log\left(f(x)_y\right)
\end{equation}$$

où $f(x)_y$ est la $y$-ième coordonnée de $f(x)$.

<small>C'est aussi pour ça qu'on aime bien utiliser le $\log\operatorname{softmax}$ : de toute façon on va vouloir calculer un $\log$ après.</small>


Ok, et le gradient ?


Ça se corse un peu mais pas trop.


On va utiliser les mêmes idées que celles qu'on a vu pour les classifieurs logistiques : on va considérer un paramètre $θ$ qui sera une concaténation de tous les poids de toutes les couches du réseau dans un gros vecteur et les les $L(f, x, y)$ comme des fonctions de $θ$.


Par bonheur, si les non-linéarités qu'on a choisi sont gentilles (et elles le sont, on les choisit pour), ces fonctions seront différentiables, c'est-à-dire qu'elles ont un gradient pour tout $(x, y)$ et on peut donc leur appliquer l'algorithme de descente de gradient stochastique.


Alors quel est le problème ?


Il y en a deux :

1. Comment on calcule ces gradients ?
2. Est-ce que l'algorithme fonctionne toujours ?


Le point 1. n'est pas un problème, les fonctions en questions peuvent être compliquées, surtout si le réseau est profond, et caculer leur gradients à la main ça peut être pénible, mais heureusement on a des programmes de calcul symbolique qui ont la gentillesse de le faire pour nous. C'est ce qu'on appelle de la **différentiation automatique** dont on va voir un exemple juste après.


Le point 2. est plus délicat en théorie : on a pas de garantie théorique que l'algo fonctionne toujours, ni même réellement d'estimation de son comportement. Mais **en pratique** ça a tendance à marcher la plupart du temps : si on applique l'algo de descente de gradient avec des hyperparamètres raisonnables et suffisament de données, on arrive à trouver des bons poids.

Un [certain](https://ruder.io/optimizing-gradient-descent/) nombre de raffinement de cet algo (que vous trouverez souvent sous le nom *SGD* pour _**S**tochastic **G**radient **D**escent_) ont été développé pour essayer que ça marche le mieux possible le plus souvent possible. Deux particulièrement notables sont l'accelération de Nesterov et l'estimation adaptative des moments ([Adam](https://arxiv.org/abs/1412.6980)).

## En pratique 🔥

En pratique, comme on ne va certainement pas implémenter tout ça à la main ici (même si je vous recommande de le faire une fois de votre côté pour bien comprendre comment ça marche), on va se reposer sur la bibliothèque de réseaux de neurones la plus utilisée pour le TAL ces dernières (et probablement aussi ces prochaines) années : [Pytorch](pytorch.org).
