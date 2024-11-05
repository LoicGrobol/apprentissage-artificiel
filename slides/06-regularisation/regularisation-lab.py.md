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
TP 6 : Régularisation
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

Les questions auxquelles vous devez répondre sont données dans des cellules tout *en italiques*. Répondez dans de nouvelles cellules (markdown et code) à la suite. Tous les coups sont permis, y compris l'utilisation d'autres packages, mais pensez à signaler lesquels dans la présente section et mettez tous vos import dans la cellule ci-dessous.

```python
import matplotlib.pyplot as plt
import numpy as np
import plotnine as p9
import polars as pl
from rich.progress import track
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
```

## Le jeu de données Iris


On le charge et on regarde ce qu'il contient.

```python
iris = datasets.load_iris()
print(iris.DESCR)
```

For evaluation purposes, we use a random split. Note that this does not do class balancing etc.

```python
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=0
)


classes = list(set(iris.target))
classes
```

### Régularisation et norme des coefficients


Pour voir ce que font les régularisations dites « *bridge* » sur un modèle linéaire, on va entraîner des classifieurs logistiques par descente de gradient stochastique et tracer les valeurs des coefficients, dont on va afficher les valeurs et les normes $L¹$ et $L²$ :

```python
# Coefs are set to zero initially: see <https://github.com/scikit-learn/scikit-learn/blob/6e9039160f0dfc3153643143af4cfdca941d2045/sklearn/linear_model/_stochastic_gradient.py#L221>
# log_loss gives us a logistic classifier
clf = SGDClassifier(
    alpha=0.1,
    eta0=1,
    learning_rate="constant",
    loss="log_loss",
    penalty=None,
)

X = X_train
y = y_train

batch_size = 8
n_epochs = 32
coefs = []

for _ in range(n_epochs):
    # we want to see the data in a different order at each epoch. Setting `random_state` and
    # reloading X and y from the dataset ensures that these shuffle are all the same accross
    # different runs of this cell.
    X, y = shuffle(X, y, random_state=0)
    for i in range(0, X.shape[0], batch_size):
        clf.partial_fit(X[i : i + batch_size], y[i : i + batch_size], classes=classes)
        coefs.append(clf.coef_.ravel().copy())

plt.figure()
plt.plot(np.arange(len(coefs)), coefs, marker=".", linestyle="")
plt.xlabel("Steps")
plt.ylabel("Coefficients")
plt.axis("tight")

plt.figure()
plt.plot(np.arange(len(coefs)), np.linalg.norm(coefs, ord=2, axis=1), marker=".", linestyle="")
plt.ylim(bottom=0)
plt.xlabel("Steps")
plt.ylabel("Coefficients $L²$ norm")

plt.figure()
plt.plot(np.arange(len(coefs)), np.linalg.norm(coefs, ord=1, axis=1), marker=".", linestyle="")
plt.ylim(bottom=0)
plt.xlabel("Steps")
plt.ylabel("Coefficients $L¹$ norm")

plt.show()
```

> *Tracez les mêmes courbes pour les régularisations $L¹$, $L²$ et elasticnet. Faites un court résumé de vos observations et interprétations.*

### Et le modèle ?


Il faut **toujours** vérifier ce que fait le modèle, et dans ce cas, manifestement, il n'est pas terrible :

```python
# Coefs are set to zero initially: see <https://github.com/scikit-learn/scikit-learn/blob/6e9039160f0dfc3153643143af4cfdca941d2045/sklearn/linear_model/_stochastic_gradient.py#L221>
clf = SGDClassifierclf = SGDClassifier(
    alpha=0.1,
    eta0=1,
    learning_rate="constant",
    loss="log_loss",
    penalty="l1",
)

X = X_train
y = y_train

batch_size = 8
n_epochs = 32
coefs = []
test_accuracies = []
train_accuracies = []

for _ in range(n_epochs):
    # we want to see the data in a different order at each epoch. Setting `random_state` and
    # reloading X and y from the dataset ensures that these shuffle are all the same accross
    # different runs of this cell.
    X, y = shuffle(X, y, random_state=0)
    for i in range(0, X.shape[0], batch_size):
        clf.partial_fit(X[i : i + batch_size], y[i : i + batch_size], classes=classes)
        coefs.append(clf.coef_.ravel().copy())
        train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))
        test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=clf.classes_)

plt.figure()
plt.plot(
    np.arange(len(train_accuracies)),
    train_accuracies,
    np.arange(len(test_accuracies)),
    test_accuracies,
    marker=".",
    linestyle="",
)
plt.ylim(bottom=0)
plt.xlabel("Steps")
plt.ylabel("Score")

plt.show()

```

> *Allez voir [la doc](https://scikit-learn.org/dev/auto_examples/linear_model/plot_sgd_iris.html), et faites une conjecture sur les raisons de cet échec. Essayez ensuite de trouver de meilleurs hyperparmètres (tous les coups sont permis). Pensez à montrer votre travail : ne donnez pas juste des valeurs mais expliquez comment vous les avez trouvées.*

## Des données plus intéressantes : 20 newsgroup

```python
twentyng = datasets.fetch_20newsgroups_vectorized()
print(twentyng.DESCR)
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    twentyng.data, twentyng.target, test_size=0.25, random_state=0
)

classes = list(set(twentyng.target))
classes
```

```python
print(X_train.shape)
```

```python
# Coefs are set to zero initially: see <https://github.com/scikit-learn/scikit-learn/blob/6e9039160f0dfc3153643143af4cfdca941d2045/sklearn/linear_model/_stochastic_gradient.py#L221>
clf = SGDClassifierclf = SGDClassifier(
    loss="log_loss",
    penalty=None,
)

X = X_train
y = y_train

batch_size = 128
n_epochs = 2
norms = []

for e in range(n_epochs):
    X, y = shuffle(X, y, random_state=0)
    # With a progress bar woo
    for i in track(range(0, X.shape[0], batch_size), description=f"Epoch {e}", transient=True):
        clf.partial_fit(X[i : i + batch_size], y[i : i + batch_size], classes=classes)
        norms.append({str(n): np.linalg.norm(clf.coef_.ravel(), ord=n) for n in [1, 2]})

y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=twentyng.target_names))

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=clf.classes_)

plt.show()

```

```python
pl.from_dicts(norms)
```
