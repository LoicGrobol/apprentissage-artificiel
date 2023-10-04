---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

<!-- #region slideshow={"slide_type": "slide"} -->
TP 3 : `scikit-learn`
=======================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

```python
from IPython.display import display
```

## scikit-learn  ?

[scikit-learn](https://scikit-learn.org/stable/index.html).

scikit-learn est une bibliothèque Python dédiée à l'apprentissage artificiel qui repose sur
[NumPy](https://numpy.org/) et [SciPy](https://scipy.org/). Il est écrit en Python et
[Cython](https://cython.org/). Il s'interface très bien avec [matplotlib](https://matplotlib.org),
[seaborn](https://seaborn.pydata.org/) ou [pandas](https://pandas.pydata.org/) (qui lui-même marche
très bien avec [plotnine](https://plotnine.readthedocs.io/)). C'est devenu un incontournable du
*machine learning* et des *data sciences* en Python.

Dans ce notebook on se limitera à la classification, une partie seulement de ce qu'offre
scikit-learn.

La classification est souvent utilisée en TAL, par exemple dans les tâches d'analyse de sentiment,
de détection d'émotion ou l'identification de la langue.

On va faire de l'apprentissage *supervisé* de classifieurs : l'idée est d'apprendre un modèle à
partir de données réparties en classes (une classe et une seule pour chaque exemple), puis de ce
servir de ce modèle pour répartir parmi les mêmes classes des données nouvelles

Dit autrement, on a un échantillon d'entraînement $\matchcal{D}$, composé de $n$ couples $(X_{i},
Y_{i}), i=1, …, n$ où les $X_{i}$ sont les entrées (en général des **vecteurs** de traits ou
*features*) et les $y_{i}$ seront les sorties, les classes à prédire. On cherche alors dans une
famille $\mathbb{M}$ de modèles un modèle de classification $M$ qui soit le plus performant possible
sur $\matchcal{D}$.

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

C'est la clé de voute du *machine learning*, vous le savez n'est-ce pas ? Nous allons travailler
avec un des jeux de données fourni par scikit-learn : [le jeu de données de reconnaissance des
vins](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)

C'est plus facile pour commencer parce que les données sont déjà nettoyées et organisées, mais vous
pourrez bien sûr par la suite [charger des données venant d'autres
sources](https://scikit-learn.org/stable/datasets/loading_other_datasets.html).

```python
from sklearn import datasets
wine = datasets.load_wine()
type(wine)
```

(La recommandation des développeureuses de `scikit-learn` est d'importer uniquement les parties qui
nous intéresse plutôt que tout le package. Notez aussi le nom `sklearn` pour l'import.)

Ces jeux de données sont des objets `sklearn.utils.Bunch`. Organisés un peu comme des dictionnaires
Python, ces objets contiennent :

- `data` : array NumPy à deux dimensions d'échantillons de données de dimensions `(n_samples,
  n_features)`, les inputs, les X
- `target` : les variables à prédire, les catégories des échantillons si vous voulez, les outputs,
  les y
- `feature_names`
- `target_names`

Et d'autres trucs comme

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

On peut convertir ces données en `DataFrame` pandas si on veut.

```python
import pandas as pd

df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df['target']=wine.target
df.head()
```

Mais l'essentiel est de retrouver nos inputs $X$ et outputs $y$ nécessaires à l'apprentissage.

```python
X_wine, y_wine = wine.data, wine.target
```

```python
X_wine.shape
```

```python
y_wine
```

Vous pouvez séparer les données en train et test facilement à l'aide de
`sklearn.model_selection.train_test_split` (voir la
[doc](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split))

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
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.title("Répartition des classes") 
plt.show()
```

Il ne faut pas hésiter à recourir à des représentations graphiques quand vous manipulez les données.
Ici on voit que la répartition des classes à prédire n'est pas homogène pour les données de test.  
On peut y remédier en utilisant le paramètre `stratify`, qui fait appel à
[`StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
pour préserver la même répartition des classes dans le train et dans le test.

```python
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.25, stratify=y_wine)
plt.hist(y_train, align="right", label="train") 
plt.hist(y_test, align="left", label="test") 
plt.legend()
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.title("Répartition des classes avec échantillonnage stratifié") 
plt.show()
```

## Entraînement

L'étape suivante est de choisir un algorithme (un *estimator* dans la terminologie de scikit-learn),
de l'entraîner sur nos données (avec la fonction `fit()`) puis de faire la prédiction (avec la
fonction `predict`).

Quelque soit l'algo choisi vous allez retrouver les fonctions `fit` et `predict`. Ce qui changera ce
seront les paramètres à passer au constructeur de la classe de l'algo. Votre travail portera sur le
choix de ces paramètres.

Exemple un peu bateau avec une méthode de type SVM.

```python
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
```

```python
clf.predict(X_test)
```

## Évaluation

On fait l'évaluation en confrontant les prédictions sur les `X_test` et les `y_test`. La fonction
`score` nous donne l'exactitude (*accuracy*) moyenne du modèle.

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


1. Refaites une partition train/test différente et comparez les résultats
2. Essayez un autre algo de classification ([un SVM à fonction de base
   radiale](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) par exemple) et
   comparez les résultats.
   - Voir [le tuto sur les noyaux
     SVM](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html) pour une idée de
     ce que ça signifie d'utiliser un RBF.

## Validation croisée

Pour améliorer la robustesse de l'évaluation on peut utiliser la validation croisée
(*cross-validation*). `scikit-learn` a des classes pour ça.

```python
from sklearn.model_selection import cross_validate, cross_val_score
print(cross_validate(LinearSVC(), X_wine, y_wine)) # infos d'accuracy mais aussi de temps
print(cross_val_score(LinearSVC(), X_wine, y_wine)) # uniquement accuracy
```

## Optimisation des hyperparamètres

L'optimisation des hyperparamètres est la dernière étape. Ici encore `scikit-learn` nous permet de
le faire de manière simple et efficace. Nous utiliserons `sklearn.model_selection.GridSearchCV` qui
fait une recherche exhaustive sur tous les paramètres donnés au constructeur. Cette classe utilise
aussi la validation croisée.

```python
from sklearn.svm import SVC
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
utilisés pour représenter les textes.

```python
from sklearn.datasets import fetch_20newsgroups

categories = [
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
]

data_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
)
```

```python
print(len(data_train.data))
print(len(data_test.data))
```

Ici on a un jeu de 2373 textes catégorisés pour train. À nous d'en extraire les features désirées.
Le modèle des [sacs de
mots](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
est le plus basique.

Attention aux valeurs par défaut des paramètres. Ici par exemple on passe tout en minuscule et la
tokenisation est rudimentaire. Ça fonctionnera mal pour d'autres langues que l'anglais. Cependant,
presque tout est modifiable et vous pouvez passer des fonctions de prétraitement personnalisées.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data) # données de train vectorisées
y_train = data_train.target
X_train.shape
```

Voilà la tête que ça a

```python
X_train[0, :]
```

Euh


La tête que ça a

```python
print(X_train[0, :])
```

```python
X_test = vectorizer.transform(data_test.data)
y_test = data_test.target
```

Pour l'entraînement et l'évaluation on reprend le code vu auparavant

```python
clf = LinearSVC(C=0.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

[TF⋅IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
est un raffinement de ce modèle, qui donne en général de meilleurs résultats.

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

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

clf = LinearSVC(C=0.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 🤖 Exo  🤖

### 1. Un projet complet

L'archive [`imdb_smol.tar.gz`](data/imdb_smol.tar.gz) (aussi disponible [dans le
dépôt](https://github.com/LoicGrobol/apprentissage-artificiel/blob/main/slides/06-scikit-learn/data/imdb_smol.tar.gz))
contient 602 critiques de films sous formes de fichiers textes, réparties en deux classes :
positives et négatives (matérialisées par des sous-dossiers). Votre mission est de réaliser un
script qui :

- Charge et vectorise ces données
- Entraîne et compare des classifieurs sur ce jeu de données

L'objectif est de déterminer quel type de vectorisation et de modèle semble le plus adapté et quels
hyperparamètres choisir. Vous pouvez par exemple tester des SVM comme ci-dessus, [un modèle de
régression
logistique](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[un arbre de
décision](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html),
[un modèle bayésien
naïf](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) ou
[une forêt d'arbres de
décision](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).


### 2. D'autres traits

Essayez avec d'autres *features* : La longueur moyenne des mots, le nombre ou le type d'adjectifs,
la présence d'entités nommées, …

Pour récupérer ce genre de *features*, vous pouvez regarder du côté de [spaCy](http://spacy.io/)
comme prétraitement de vos données.
