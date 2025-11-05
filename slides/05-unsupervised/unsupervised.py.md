---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: cours-ml
    language: python
    name: python3
---

Apprentissage non-supervisé
===========================

**L. Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

Ce TP est en grande partie inspiré d'éléments de la documentation de scikit-learn et umap-learn.
Vous devriez allez les lire en détails pour en savoir plus.


Avant de commencer, allez jouer un peu avec [une version interactive de
K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) (ou cherchez en
d'autres !) pour se rafraîchir les idées.


Le sujet est vraiment vaste et on ne va évidemment pas être exhaustifves, mais essayons de donner
quelques idées. D'abord quelques outils.

```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

Voilà une façon de générer et afficher des données aléatoires en 2d.

```python
# On récupère un générateur de nombres pseudo-aléatoires, en fixant la seed pour que les résultats
# soient reproductibles (si vous voulez un autre échantillon, changez la seed)
gen = np.random.default_rng(0)
# On génère un array de flottants aléatoires compris entre -8.0 et 8.0 distribués uniformément, de
# taille 1024×2 (autrement dit l'équivalent de 1024 points). Oui j'ai une obsession avec les
# puissances de 2, demandez-moi pourquoi 10 minutes avant la fin du cours
points = gen.uniform(-8.0, 8.0, size=(1024, 2))
# On récupère des objets pour dessiner avec pyplots. Dans ce cas simple on pourrait s'en passer,
# mais ça sera une habitude utile quand on voudra faire des graphiques plus complexes.
fig, ax = plt.subplots()
# Le plot proprement dit. Le paramètre marker est un moyen rapide d'avoir des petits points.
ax.scatter(x=points[:, 0], y=points[:, 1], marker=".")
fig.show()
```

Notez que dans la vie, pour faire des jolis graphiques sérieusement, il vaut mieux utiliser quelque
chose comme [plotnine](https://plotnine.org/), mais ici on va rester sur pyplot, qui est un peu
rudimentaire mais va nous suffire.


Maintenant si on a un tableau qui associe une classe à chaque point, on peut les visualiser avec des
couleurs comme ça (attention, il faut que les entrées soient des entiers)

```python
gen = np.random.default_rng(0)

points = gen.uniform(-8.0, 8.0, size=(1024, 2))

# Ici on leur donne aléatoirement une classe, soit 0 soit 1
clusters = gen.integers(0, 2, size=(1024,))

fig, ax = plt.subplots()
ax.scatter(x=points[:, 0], y=points[:, 1], c=clusters, marker=".")
fig.show()

```

Si vous trouvez les couleurs moches, allez voir [la
doc](https://matplotlib.org/stable/gallery/color/colormap_reference.html). Notez que les points sont
au même endroit que sur le graphique précédent : normal, on a la même seed :)


scikit-learn a aussi des fonctions pour générer des points avec des propriétés choisies, comme
[`make_blobs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) :

```python
# On lui passe directement le nombre de point, la dimension, une seed, le nombre de blobs qu'on veut faire.
points, _y = make_blobs(n_samples=1024, n_features=2, centers=3, random_state=0)

fig, ax = plt.subplots()
ax.scatter(x=points[:, 0], y=points[:, 1], marker=".")
fig.show()
```

**Entraînement** : allez lire la doc de de `make_blobs` pour comprendre comment elle marche, puis
modifiez la cellule ci-dessous pour que les blobs soient coloriés, et que leurs centres soient
générés dans la même *bounding box* que nos points aléatoires du début.

Au fait, pourquoi j'avais mis un underscore dans `_y` ?

```python
points, _y = make_blobs(n_samples=1024, n_features=2, centers=3, random_state=0)

fig, ax = plt.subplots()
ax.scatter(x=points[:, 0], y=points[:, 1], marker=".")
fig.show()

```

Au fait, pourquoi on fait des jeux de données aléatoires (une forme de données *synthétiques*) ?
Parce qu'en général les jeux de données réels ont plus de deux dimensions (même le tout petit
[Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) en a quatre) et c'est
donc plus difficile à visualiser directement. On s'oocupera de leurs cas plus tard.

## Zoologie des algos de clustering


Comme toujours, on va commencer par un peu de zoologie : on va regarder quelques algos de clustering
bien connus sur des données qu'on maîtrise, pour voir ce qui se passe. Voici par exemple comment
trouver des clusters avec K-Means sur un dataset de blobs :

```python
points, _y = make_blobs(n_samples=1024, n_features=2, centers=3, random_state=0)

# On créée l'estimateur
kmeans = KMeans(n_clusters=3, random_state=0)
# On le fit sur nos données
clusters = kmeans.fit_predict(points)

fig, ax = plt.subplots()
ax.scatter(x=points[:, 0], y=points[:, 1], c=clusters, marker=".")
fig.show()

```

Ça a l'air assez cohérent, est-ce que ça colle à nos données ? On peut vérifier en
[affichant](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#visualize-the-results-on-pca-reduced-data)
les blobs originaux et les centroïdes par dessus le diagramme de Voronoi :

```python
points, blobs = make_blobs(n_samples=1024, n_features=2, centers=3, random_state=0)

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(points)

# Un peu de travail pour afficher le diagramme de Voronoi.
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
grid, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(grid.shape)

fig, ax = plt.subplots()
ax.imshow(
    z,
    interpolation="nearest",
    extent=(grid.min(), grid.max(), yy.min(), yy.max()),
    aspect="auto",
    origin="lower",
)

# Plot the original points with their original blob
ax.scatter(x=points[:, 0], y=points[:, 1], c=blobs, marker=".")
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
fig.show()

```

Est-ce que c'est un bon clustering du coup ? Réflechissez à la question et on en débat toustes
ensemble tout à l'heure?


⚠️ **Attention** un diagramme de Voronoi, ça n'a évidemment de sens que pour k-means et ses dérivés
(et éventuellement des trucs comme BGMM), ça n'en a par exemple pas du tout pour du clustering
hiérarchique.


**Entraînement** Testez quelques trucs :

- Clusteriser avec un $k$ plus grand ou plus petit que le nombre de blobs.
- Clusteriser des points générés uniformément (donc où il n'y a pas de blobs a priori et donc pas de
  raisons de trouver des clusters en principe)

Faites vos tests à la suite de cette cellule, prenez des notes, appellez moi pour des questions,
gardez votre travail précieusement, ça vous servira plus tard.

### À vous de jouer


Examinez les comportements des algos suivants :

- [Clustering Agglomératif](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering)
- [Mélanges de Gaussiennes Bayésiennes](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
- BIRCH
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - [Y a une visu!](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

Testez-les sur

- Des blobs (en faisant varier les paramètres).
- De points générés uniformément.
- [Des cercles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
- [Des lunes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.htm)

Toujours pareil : prenez des notes, faites des graphiques, lisez la doc.

Une fois que vous avez fait ça sérieusement et seulement à ce moment là, allez voir [la gallerie de
scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
pour une vision d'ensemble (si vous commencez par ça, ça vous apportera pas grand chose).
