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

Et la troisième couchec comme

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
