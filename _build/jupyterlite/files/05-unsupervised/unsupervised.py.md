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
      jupytext_version: 1.18.1
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
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
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
# On lui passe directement le nombre de point, la dimension, une seed, le nombre de blobs qu'on veut
# faire.
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

# On créée l'estimateur
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

## Réduction de dimension

Comme on l'a dit précédemment, les données du monde réel ont la fâcheuse habitude de ne pas être
entre 3 dimensions, et encore moins en deux. Même des données jouets :

- [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) → **4**
- [Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) →
  **13**
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database) images des 28×28 pixels → **784**
  - Même chose pour [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) évidemment
- [20ng
  vectorisé](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized)
  → **130107**!
- …

Ça pose deux problèmes :

- Ça rend ces données impossibles à visualiser : même si votre cerveau arrivait à penser en n
  dimensions, il reste que vous vivez dans un monde à 3 dimensions, avec des organes sensoriels à 2
  dimensions, impossible de représenter plus.
- D'un point de vue computationnel, pour n'importe quelle manipulation qu'on veut faire sur ces
  donnés (apprentissage ou pas, supervisé ou pas)
  - Plus on a de dimensions, plus les calculs sont complexes.
  - Plus on a de dimensions, plus la stabilité des approximations qu'on fait est fragile, pour les
    nombreuses raisons qu'on appelle parfois [« fléau de la
    dimension »](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

De plus, en général, les dimensions de nos données ne sont en général pas indépendantes (imaginer
pourquoi sur chacun des exemples précédents). D'un point de vue mathématique, le support de nos
données a souvent une dimension intrinsèque bien inférieure à celle de l'espace ambiant. En
conséquence, non seulement les hautes dimensions nous compliquent l'existence, mais souvent avec peu
de valeur ajoutée.

⚠️ Ce n'est évidemment pas vrai pour des représentations de données qui ont été explicitement pour
prévu pour ça, par exemple les représentations vectorielles denses (ou *embeddings*) dont on
reparlera.

On va donc souvent pouvoir réduire ses problèmes avec des techniques de réduction de dimension, dont
on peut résumer l'idée par :

> Étant donné une famille de vecteurs $E = (x_1, …, x_N)$ de $ℝ^d$, on cherche une transformation
> $T:ℝ^d\longrightarrow ℝ^{d'}$ avec $d'<d$ telle que $E'=(T(x_1), …, T(x_n))$ préserve au moins
> approximativement certaines des structures de $E$.

En général les structure en question seront de même nature celles sur lesquelles on s'appuie pour
le clustering : spatiales, métriques, topologiques… Pour celà, on réalise souvent une optimisation
(sous contraintes) spécifique : on cherche $T$ telle que la variance de $E'$ soit la plus proche de
celle de $E$ (ACP), que la distribution des similarités des paires de $E$ soit la plus proche
possible de celle de $E'$ (t-SNE)…

Si on a bien fait notre job, on se retrouve avec un jeu de données plus facile à manipuler, et si
$d'⩽3$, il sera même probablement *visualisable*. Youpi.

En revanche **attention** à part dans des cas très contraints (par exemple si une dimension est
constante), une réduction de dimension va détruire de l'information (aucun de nos algos n'est
*bijectif*), on peut réduire la dimension mais pas la réaugmenter (en tout cas pas dans ce cadre).
Ça aura nécéssairement une conséquence (pas forcémenet toujours négative cependant) sur les
manipulations que vous appliquez aux données transformées.

### ACP

Commencez par [visualiser](https://setosa.io/ev/principal-component-analysis/) de quoi il s'agit.


Comme d'habitude, avec scikit-learn, c'est pas très sorcier, on peut juste utiliser [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) :

```python
# Si on a des points
points, blobs = make_blobs(n_samples=32, n_features=5, centers=5, random_state=0)
print("Points:", points)

# Il suffit d'instancier l'estimateur en disant combien de dimesion on veut garder
mon_super_pca = PCA(n_components=4)

# On trouve la décomposion et on transforme nos points
points_reduced = mon_super_pca.fit_transform(points)

print("Réduits:", points_reduced)
```

On est donc passé de 5 dimensions à 4. Évidemment tout seul ça ne nous est pas forcément très utile.
Mais voyons ce qui se passe avec des dimensions qu'on peut visualiser.


Recette :comment faire un plot en 3d (avec `%matplotlib widget` pour le rendre interactif dans jupyter).

```python
%matplotlib widget

points, blobs = make_blobs(n_samples=1024, n_features=3, centers=3, random_state=0)

fig = plt.figure()
ax = fig.add_subplot(aspect="equal", projection="3d")

ax.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], c=blobs, marker=".")
# Pour une raison inconnue, fig.show() affiche deux fois la figure, je déteste un peu matplotlib
plt.show()
```

Maintenant voyons l'effet d'une ACP

```python
%matplotlib widget

points, blobs = make_blobs(n_samples=1024, n_features=3, centers=3, random_state=0)
pca = PCA(n_components=2)
points_proj = pca.fit_transform(points)  # ← toute ce qui est important se passe ici

fig = plt.figure()
ax3d = fig.add_subplot(1, 2, 1, aspect="equal", projection="3d")
ax2d = fig.add_subplot(1, 2, 2, aspect="equal")

ax3d.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], c=blobs, marker=".")
ax2d.scatter(x=points_proj[:, 0], y=points_proj[:, 1], c=blobs, marker=".")

# Show projection axes
print(pca.components_)
ax3d.plot(
    [0.0, 4 * pca.components_[0, 0]],
    [0.0, 4 * pca.components_[0, 1]],
    [0.0, 4 * pca.components_[0, 2]],
    c="red",
)
ax3d.plot(
    [0.0, 4 * pca.components_[1, 0]],
    [0.0, 4 * pca.components_[1, 1]],
    [0.0, 4 * pca.components_[1, 2]],
    c="blue",
)

ax2d.plot(
    [0.0, 4.0],
    [0.0, 0.0],
    c="red",
)
ax2d.plot(
    [0.0, 0.0],
    [0.0, 4.0],
    c="blue",
)
plt.show()
```

Évidemment, comme d'habitude toujours, scikit-learn propose une implémentation avec toutes les
astuces possibles pour que ça se passe bien, et vous donne le contrôle dessus avec plein
d'hyperparamètres dont les valeurs par défaut sont généralement pas mal, mais qu'il est bon d'aller
chercher à comprendre.


Astuce : si vous voulez les valeurs propres de la matrice de covariance (donc les axes dans l'espace
de départ), comme vous le voyez ci-dessus, vous pouvez les récupérer apreès `fit` dans
`pca.components_`.

#### 📡 Entraînement 📡

À vous de jouer ! Utilisez des ACP pour visualiser les classes des datasets Iris et Wine et voyez si
ça marche bien avec leurs features.


- Exo: calculer les variances suivant les axes et faire un plot variance = f(d')

### t-SNE

Disponible comme
[sklearn.manifold.TSNE](scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), voir
aussi [OpenTSNE](https://opentsne.readthedocs.io) pour une implémentation plus agréable par certains
aspects (notamment une méthode `transform` qui marche), allez voir leur
[démo](https://opentsne.readthedocs.io/en/stable/examples/01_simple_usage/01_simple_usage.html).

```python
%matplotlib widget

points, blobs = make_blobs(n_samples=1024, n_features=3, centers=3, random_state=0)
tsne = TSNE(n_components=2)
points_proj = tsne.fit_transform(points)  # ← toute ce qui est important se passe ici

fig = plt.figure()
ax3d = fig.add_subplot(1, 2, 1, aspect="equal", projection="3d")
ax2d = fig.add_subplot(1, 2, 2, aspect="equal")

ax3d.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], c=blobs, marker=".")
ax2d.scatter(x=points_proj[:, 0], y=points_proj[:, 1], c=blobs, marker=".")

# Pas d'axes de projection, c'est pas comme ça que marche t-SNE!
plt.show()

```

Et pour Iris

```python
%matplotlib widget


features, species = load_iris(return_X_y=True)
tsne = TSNE(n_components=2)
data_proj = tsne.fit_transform(features)  # ← toute ce qui est important se passe ici

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(data_proj[:, 0], data_proj[:, 1], c=species, marker=".")

plt.show()

```

Est-ce que c'est mieux qu'une ACP ? Peut-être un peu mais ça a peu de sens pour des données aussi petites.

#### 🪻 Entraînement 🪻

Essayez en 3d, puis comparez les résultats obtenus en

- Faisant un t-SNE en 3D et en ne gardant que les deux premières coordonnées
- Faisant directement un t-SNE en 2D

Pour une ACP ça donne évidemment le même résultat, mais pour t-SNE il n'y a pas de raison que ce soit le cas !

#### 🍷 Entraînement 🍷

Essayez sur Wine et MNIST (`sklearn.datasets.load_digits`) pour voir.

#### 🤔


Essayons d'appliquer t-SNE à des données qui n'ont pas de raison d'avoir une structure :

```python
%matplotlib widget

gen = np.random.default_rng(0)
points = gen.uniform(-8.0, 8.0, size=(1024, 8))
# Un truc aléatoire mais avec une vague forme
points = points*np.arange(points.shape[1])
tsne = TSNE(n_components=2)
points_proj = tsne.fit_transform(points)  # ← toute ce qui est important se passe ici

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(points_proj[:, 0], points_proj[:, 1], marker=".")

plt.show()

```

Qu'est-ce que vous en pensez ? Appellez moi qu'on en discute ensemble.


Autre point à noter, qui est dit assez clairement dans les docs et les articles sur t-SNE mais que
les gens oublient souvent : t-SNE galère en très haute dimension, ça peut donc valoir le coup de
faire d'abord une ACP pour passer les données en quelque chose comme 50 dimensions pour ensuite
appliquer t-SNE dans cet espace moins complexe.

### UMAP

Un des derniers variants de la famille de t-SNE (en tout cas basé sur les mêmes principes, mais en
mieux). C'est celui que je vous conseille. Il n'est pas dans scikit-learn (pour l'instant ?), mais
il a un package bien fait : [`umap-learn`](https://umap-learn.readthedocs.io).

```python
%matplotlib widget

points, blobs = make_blobs(n_samples=1024, n_features=3, centers=3, random_state=0)
# Rappel: je mets les imports en début de notebook et vous devriez aussi
umap = UMAP(n_components=2)
points_proj = umap.fit_transform(points)

fig = plt.figure()
ax3d = fig.add_subplot(1, 2, 1, aspect="equal", projection="3d")
ax2d = fig.add_subplot(1, 2, 2, aspect="equal")

ax3d.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], c=blobs, marker=".")
ax2d.scatter(x=points_proj[:, 0], y=points_proj[:, 1], c=blobs, marker=".")

plt.show()

```

Voyons sur MNIST

```python
%matplotlib widget

points, digits = load_digits(return_X_y=True)
# Rappel: je mets les imports en début de notebook et vous devriez aussi
umap = UMAP(n_components=2)
points_proj = umap.fit_transform(points)

fig = plt.figure()
ax = fig.add_subplot(aspect="equal")

sc = ax.scatter(x=points_proj[:, 0], y=points_proj[:, 1], c=digits, marker=".")

# Ceci pour faire une légende. Je vous ai déjà dit que je déteste pyplot ?
handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o")[0] for c in range(10)]
labels = list(range(10))
ax.legend(handles, labels)

plt.show()

```

#### 👗 Entraînement 👗

- Voir ce que ça donne en 3D
- Tester aussi sur [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - Téléchargez les données et chargez les à la mano, sans leur `mnist_reader`, ça vous apprendra.
