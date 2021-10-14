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
Cours 7 : `scikit-learn`
=======================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-10-05
<!-- #endregion -->

```python
from IPython.display import display
```

## `scikit-learn ` ?

[`scikit-learn `](https://scikit-learn.org/stable/index.html).

`scikit-learn ` est une bibliothèque Python dédiée à l'apprentissage artificiel. Ce package est
développé sur licence libre. Il y a une forte proportion de français parmi les développeurs, le
projet est soutenu par l'INRIA notamment.

`scikit-learn ` repose sur NumPy et SciPy. Il est écrit en Python et Cython. Il s'interface très
bien avec `matplotlib`, `plotly` ou `pandas`. C'est devenu un incontournable du *machine learning*
et de la *datascience* en Python.

Dans ce notebook nous nous limiterons à la classification, une partie seulement du package [scikit-learn](https://scikit-learn.org/stable/index.html).

La classification est souvent utilisée en TAL, par exemple dans les tâches d'analyse de sentiment,
de détection d'émotion ou l'identification de la langue.

On va faire de l'apprentissage *supervisé*, vous connaissez la chanson : l'idée est d'apprendre un
modèle à partir de données étiquetées et de prédire la bonne étiquette pour une donnée inconnue du
modèle.

Dit autrement, on a un échantillon d'entraînement composé de $n$ couples $Z_{i}=(X_{i}, Y_{i}),
i=1...n$ où les $X_{i}$ sont les inputs avec plusieurs traits et les $Y_{i}$ seront les outputs, les
catégories à prédire.

L'objectif du problème d'apprentissage est de trouver une fonction $g:X→Y$ de prédiction, qui
minimise les erreurs de prédiction.

`scikit-learn` offre beaucoup d'algorithmes d'apprentissage. Vous en trouverez un aperçu sur
[cette carte](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) et sur ces
listes : [supervisé](https://scikit-learn.org/stable/supervised_learning.html) / [non
supervisé](https://scikit-learn.org/stable/unsupervised_learning.html).

Mais `scikit-learn` offre également les outils pour mener à bien les étapes d'une tâche de
d'apprentissage :

- Manipuler les données, constituer un jeu de données d'entraînement et de test
- Entraînement du modèle
- Évaluation
- Optimisation des hyperparamètres

```python
%pip install -U scikit-learn
```

## Un premier exemple

### Les données

C'est la clé de voute du *machine learning*, vous le savez n'est-ce pas ? Nous allons travailler
avec un des jeux de données fourni par scikit-learn : [le jeu de données de reconnaissance des vins](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)

C'est plus facile pour commencer parce que les données sont déjà nettoyées et organisées.

```python
from sklearn import datasets
wine = datasets.load_wine()
type(wine)
```

(La recommandation des développeureuses de `scikit-learn` est d'importer uniquement les parties qui
nous intéresse plutôt que tout le package. Notez aussi le nom `sklearn` pour l'import.)

Ces jeux de données sont des objets `sklearn.utils.Bunch`. Organisés un peu comme des dictionnaires
Python, ces objets contiennent :

- `data` : array NumPy à deux dimensions d'échantillons de données (n_samples * n_features), les
  inputs, les X
- `target` : les variables à prédire, les catégories des échantillons si vous voulez, les outputs,
  les y
- `feature_names` 
- `target_names`

```python
print(wine.DESCR)
```

```python
wine.feature_names
```

Si on a installé `pandas`

```python
%pip install -U pandas
```

On peut convertir ces données en `Dataframe` pandas si on veut.

```python
import pandas as pd

df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df['target']=wine.target
df.head()
```

Mais l'essentiel est de retrouver nos inputs X et outputs y nécessaires à l'apprentissage.

```python
X_wine, y_wine = wine.data, wine.target
```

Vous pouvez séparer les données en train et test facilement à l'aide de `sklearn.model_selection.train_test_split` ( voir la [doc](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split))

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3)
y_train
```

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.hist(y_train, align="right", label="train") 
plt.hist(y_test, align="left", label="test")
plt.legend()
plt.xlabel("classe")
plt.ylabel("nombre d'exemples")
plt.title("répartition des classes") 
plt.show()
```

Il ne faut pas hésiter à recourir à des représentations graphiques quand vous manipulez les données.
Ici on voit que la répartition des classes à prédire n'est pas homogène pour les données de test.  
On peut y remédier en utilisant le paramètre `stratify`, qui fait appel à [`StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) pour préserver la même répartition des classes dans le train et dans le test.

```python
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.25, stratify=y_wine)
plt.hist(y_train, align="right", label="train") 
plt.hist(y_test, align="left", label="test") 
plt.legend()
plt.xlabel("classe")
plt.ylabel("nombre d'exemples")
plt.title("répartition des classes avec échantillonnage stratifié") 
plt.show()
```

## Entraînement

L'étape suivante est de choisir un algorithme (un *estimator*), de l'entraîner sur nos données train
(avec la fonction `fit()`) puis de faire la prédiction (avec la fonction `predict`).  
Quelque soit l'algo choisi vous allez retrouver les fonctions `fit` et `predict`. Ce qui changera ce
seront les paramètres à passer au constructeur de la classe de l'algo. Votre travail portera sur le
choix de ces paramètres.

Exemple un peu bateau avec une méthode de type SVM.

```python
from sklearn.svm import SVC
clf = SVC(C=1, kernel="linear")
clf.fit(X_train, y_train)
```

```python
clf.predict(X_test)
```

## Évaluation

On fait l'évaluation en confrontant les prédictions sur les `X_test` et les `y_test`. La fonction `score` nous donne l'exactitude (*accuracy*) moyenne du modèle.

```python
clf.score(X_test, y_test)
```

Pour la classification il existe une classe bien pratique : `sklearn.metrics.classification_report`

```python
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## ✍️ Exo ✍️


1. Essayez un autre algo de classification (Un SVM polynomial par exemple) et comparez les résultats.
2. Sur ce même algo, refaites une partition train/test et comparez l'évaluation avec les résultats
   précédents. 

## Validation croisée

Pour améliorer la robustesse de l'évaluation on va utiliser la validation croisé
(*cross-validation*). `scikit-learn` a des classes pour ça. 

```python
from sklearn.model_selection import cross_validate, cross_val_score
print(cross_validate(SVC(C=1, kernel="linear"), X_wine, y_wine)) # infos d'accuracy mais aussi de temps
print(cross_val_score(SVC(C=1, kernel="linear"), X_wine, y_wine)) # uniquement accuracy
```

## Optimisation des hyperparamètres

L'optimisation des hyperparamètres est la dernière étape. Ici encore `scikit-learn` nous permet de
le faire de manière simple et efficace. Nous utiliserons `sklearn.model_selection.GridSearchCV` qui
fait une recherche exhaustive sur tous les paramètres donnés au constructeur. Cette classe utilise
aussi la validation croisée.

```python
from sklearn.model_selection import GridSearchCV

param_grid =  {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel':['rbf','linear']}
grid = GridSearchCV(SVC(), param_grid, cv = 5, scoring = 'accuracy')
estimator = grid.fit(X_wine, y_wine)
estimator.cv_results_
```

```python
df = pd.DataFrame(estimator.cv_results_)
df.sort_values('rank_test_score')
```

## Classification de textes

Le [dataset 20
newsgroups](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
est un exemple de classification de textes proposé par `scikit-learn`. Il y a aussi [de la
doc](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) sur
les traits (*features*) des documents textuels.

La classification avec des techniques non neuronales repose en grande partie sur les traits
utilisées pour représenter les textes.

```python
from sklearn.datasets import fetch_20newsgroups

categories = [ 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space']

data_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
)

data_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
)
```

```python
print(len(data_train.data))
print(len(data_test.data))
```

Ici on a un jeu de 2373 textes catégorisés pour train. À nous d'en extraire les features désirées.
tf⋅idf est un grand classique.

Attention aux valeurs par défaut des paramètres. Ici par exemple on passe tout en minuscule et la
tokenisation est rudimentaire. Ça fonctionnera mal pour d'autres langues que l'anglais.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english'
)
X_train = vectorizer.fit_transform(data_train.data) # données de train vectorisées
y_train = data_train.target
X_train.shape
```

```python
X_test = vectorizer.transform(data_test.data)
y_test = data_test.target
```

Pour l'entraînement et l'évaluation on reprend le code vu auparavant

```python
clf = SVC(C=1, kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 🤖 Exo  🤖

### 1. D'autres traits
Essayez avec d'autres *features* : La longueur moyenne des mots, le nombre d'adjectifs, la présence
d'entités nommées, …

Pour récupérer ce genre de features, vous pouvez regarder du côté de [spaCy](http://spacy.io/).


### 2. Et les réseaux de neurones ?

`scikit-learn` permet d'utiliser un Multi-layer Perceptron (MLP). Et comme la bibliothèque ne permet
pas d'utiliser un GPU pour les calculs, son utilisation est limitée à des jeux de données de taille
moyenne.

`scikit-learn` n'est pas fait pour le *deep learning*. Il existe des bibliothèques associées qui
permettent de combiner Keras ou pytorch avec `scikitlearn` néanmoins.

Essayez en suivant [la
doc](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)


Il y a encore plein d'autre choses marrantes à faire avec `scikit-learn` et on en verra, mais en attendant vous pouvez aller voir [leur exemple sur ce dataset](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py) qui a de bien jolis graphiques.
