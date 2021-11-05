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

![](https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png)

Un modèle de neurone biologique (plutôt sensoriel) : une unité qui reçoit plusieurs entrées $x_i$
scalaires (des nombres quoi), en calcule une somme pondérée $z$ (avec des poids $w_i$ prédéfinis) et
renvoie une sortie binaire $y$ ($1$ si $z$ est positif, $0$ sinon).


Autrement dit

$$\begin{align}
z &= \sum_i w_ix_i\\
y &=
    \begin{cases}
        1 & \text{si $z ≥ 0$}\\
        0 & \text{sinon}
    \end{cases}
\end{align}$$

Formulé célèbrement par McCulloch et Pitts (1943) avec des notations différentes


![](figures/perceptron/perceptron.svg)


Implémenté comme une machine, le perceptron Mark I, par Rosenblatt (1958) :

[![Une photographie en noir et blanc d'une machine ressemblant à une grande armoire pleine de fils
électriques](https://upload.wikimedia.org/wikipedia/en/5/52/Mark_I_perceptron.jpeg)](https://en.wikipedia.org/wiki/File:Mark_I_perceptron.jpeg)


**Est-ce que ça vous rappelle quelque chose ?**


🤔


C'est un **classifieur linéaire** dont on a déjà parlé [précédemment](../lecture10/lecture10.md).


Les ambitions initiales étaient grandes

> *the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence.*  
> New York Times, rappporté par Olazaran (1996)


En réalité on se heurte vite à des problèmes, même pour représenter les fonctions logiques les plus basique, comme $\operatorname{XOR}$ définie pour $x ∈ \{0, 1\}$ et  $y ∈ \{0, 1\}$ par :


$$\begin{equation}
    \operatorname{XOR}(x, y) = 
        \begin{cases}
            1 & \text{si $x ≠ y$}\\
            0 & \text{si $x = y$}
        \end{cases}
\end{equation}$$


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
