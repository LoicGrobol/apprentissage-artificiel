```python
from sklearn.datasets import load_iris

features, species = load_iris(return_X_y=True)

# On prend 3 composantes et si on en veut que deux ben on prend pas la dernière coordonnée

pca = PCA(n_components=3)
features_3d = pca.fit_transform(features)
features_2d = features_3d[:, 0:-1]

fig = plt.figure()
ax3d = fig.add_subplot(1, 2, 1, aspect="equal", projection="3d")
ax2d = fig.add_subplot(1, 2, 2, aspect="equal")

ax3d.scatter(xs=features_3d[:, 0], ys=features_3d[:, 1], zs=features_3d[:, 2], c=species, marker=".")
ax2d.scatter(x=features_2d[:, 0], y=features_2d[:, 1], c=species, marker=".")
```