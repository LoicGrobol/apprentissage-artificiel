---
title: Apprentissage artificiel — M2 PluriTAL 2024
layout: default
---

<!-- LTeX: language=fr -->

## News

- **2024-09-17** Premier cours du semestre le 18/09/2024

## Infos pratiques

- **Quoi** « Apprentissage artificiel »
- **Où** Salle 410, BFC
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 18/09 au 13/11
  - Voir le planning pour les dates exactes
- **Contact** Loïc Grobol [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)
- **Évaluation** Un TP noté en temps limité (date à déterminer) et un projet

## Liens utiles

- Prendre rendez-vous pour des *office hours* en visio :
  [mon calendrier](https://calendar.app.google/N9oW2c9BzhXsWrrv9)
- Lien Binder de secours :
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LoicGrobol/apprentissage-artificiel/main)

## Séances

Tous les supports sont sur [github](https://github.com/loicgrobol/apprentissage-artificiel), voir
[Utilisation en local](#utilisation-en-local) pour les utiliser sur votre machine comme des
notebooks. À défaut, ce sont des fichiers Markdown assez standards, qui devraient se visualiser
correctement sur la plupart des plateformes (mais ne seront pas dynamiques).

Les notebooks ci-dessous ont tous des liens Binder pour une utilisation interactive
sans rien installer.

### 2024-09-18 : Outils de travail

- {% notebook_badges slides/01-tools/python_crash_course.py.md %}
  [Crash course Python](slides/01-tools/python_crash_course.py.ipynb)
  - {% notebook_badges slides/01-tools/python_crash_course-solutions.py.md %}
    [Solutions](slides/01-tools/python_crash_course-solutions.py.ipynb)
- {% notebook_badges slides/01-tools/numpy-slides.py.md %}
  [Présentation Numpy](slides/01-tools/numpy-slides.py.ipynb)
- {% notebook_badges slides/01-tools/polars.py.md %}
  [Présentation Polars](slides/01-tools/polars.py.ipynb)

### 2024-09-25 : Intuitions et vocabulaire

- [Slides intro](slides/02-intro/intro.pdf)
- {% notebook_badges slides/02-scikit-learn/scikit-learn-slides.py.md %}
  [TP intro à scikit-learn](slides/02-scikit-learn/scikit-learn-slides.py.ipynb)
  - [`imdb_smol.tar.gz`](slides/02-scikit-learn/data/imdb_smol.tar.gz)

### 2024-10-02 : Évaluation

TP : full autonomie ! Préparez un notebook (avec du code **et** du texte) où vous utiliserez le
scikit-learn sur le jeu de données [20
newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html)
pour étudier les capacités de quelques classifieurs.

- Présenter rapidement le jeu de données (origine, composition, quelques stats descriptives)
- Entraîner [des arbres de
  décision](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
  en faisant varier la profondeur pour étudier dans quelle mesure ils sous-apprennent quand la
  profondeur est trop faible et déterminer à partir de quand ils surapprennent. Penser à la
  validation croisée, penser à faire des courbes d'apprentissage, penser à visualiser les arbres,
  penser à regarder les matrices de confusion, chercher dans [les métriques de
  scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
  celles qui pourraient être intéressantes.
- Étudier l'influence de la taille du corpus d'apprentissage (mêmes indices)
- Même jeu avec un
  [perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html),
  en étudiant l'influence du paramètre `max_iter` (commencer à 1)

Envoyer à `lgrobol@parisnanterre.fr` avec en objet `[ML2024] TP Overfit`.


### 2024-10-09 : Régression logistique et descente de gradient

- {% notebook_badges slides/04-logistic-regression/lr-slides.py.md %}
  [TP régression logistique](slides/04-logistic-regression/lr-slides.py.ipynb)
  - [Lexique VADER](slides/04-logistic-regression/data/vader_lexicon.txt)
  - [Jeu de données IMDB smol](slides/04-logistic-regression/data/imdb_smol.tar.gz)

## Utilisation en local

Les supports de ce cours sont écrits en Markdown, convertis en notebooks avec
[Jupytext](https://github.com/mwouts/jupytext). C'est entre autres une façon d'avoir un historique
git propre, malheureusement ça signifie que pour les ouvrir en local, il faut installer les
extensions adéquates. Le plus simple est le suivant

1. Récupérez le dossier du cours, soit en téléchargeant et décompressant
   [l'archive](https://github.com/LoicGrobol/apprentissage-artificiel/archive/refs/heads/main.zip)
   soit en le clonant avec git : `git clone
   https://github.com/LoicGrobol/apprentissage-artificiel.git` et placez-vous dans ce dossier.
2. Créez un environnement virtuel pour le cours (par exemple ici avec [virtualenv](https://virtualenv.pypa.io)) 

   ```console
   python3 -m virtualenv .venv
   source .venv/bin/activate
   ```

3. Installez les dépendances

   ```console
   pip install -U -r requirements.txt
   ```

4. Lancez Jupyter

   ```console
   jupyter notebook
   ```

   JupyterLab est aussi utilisable.

## Ressources

### Apprentissage artificiel

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et
  James H. Martin est **indispensable**. Il parle de bien plus de chose que simplement de
  l'apprentissage artificiel, mais sur le plan théorique il contient tout ce dont on parlera
  concernant l'apprentissage pour le TAL. Il est disponible gratuitement, donc n'hésitez pas à le
  consulter très fréquemment. J'essaierai d'indiquer pour chaque cours les chapitres en rapport.
- [*Apprentissage artificiel - Concepts et
  algorithmes*](https://www.eyrolles.com/Informatique/Livre/apprentissage-artificiel-9782416001048/)
  d'Antoine Cornuéjols et Laurent Miclet. Plus ancien mais en français et une référence très
  complète sur l'apprentissage (en particulier non-neuronal). Il est un peu cher alors si vous
  voulez l'utiliser, commencez par me demander et je vous prêterai le mien.

### Python général

Il y a beaucoup, beaucoup de ressources disponibles pour apprendre Python. Ce qui suit n'est qu'une
sélection.

#### Livres

Ils commencent à dater un peu, les derniers gadget de Python n'y seront donc pas, mais leur lecture
reste très enrichissante (les algos, ça ne vieillit jamais vraiment).

- *How to think like a computer scientist*, de Jeffrey Elkner, Allen B. Downey, and Chris Meyers.
  Vous pouvez l'acheter. Vous pouvez aussi le lire
  [ici](http://openbookproject.net/thinkcs/python/english3e/)
- *Dive into Python*, by Mark Pilgrim. [Ici](http://www.diveintopython3.net/) vous pouvez le lire ou
  télécharger le pdf.
- *Learning Python*, by Mark Lutz.
- *Beginning Python*, by Magnus Lie Hetland.
- *Python Algorithms: Mastering Basic Algorithms in the Python Language*, par Magnus Lie Hetland.
  Peut-être un peu costaud pour des débutants.
- Programmation Efficace. Les 128 Algorithmes Qu'Il Faut Avoir Compris et Codés en Python au Cours
  de sa Vie, by Christoph Dürr and Jill-Jênn Vie. Si le cours vous paraît trop facile. Le code
  Python est clair, les difficultés sont commentées. Les algos sont très costauds.

#### Web

Il vous est vivement conseillé d'utiliser un (ou plus) des sites et tutoriels ci-dessous.

- **[Real Python](https://realpython.com), des cours et des tutoriels souvent de très bonne qualité
  et pour tous niveaux.**
- [Un bon tuto NumPy](https://cs231n.github.io/python-numpy-tutorial/) qui va de A à Z.
- [Coding Game](https://www.codingame.com/home). Vous le retrouverez dans les exercices
  hebdomadaires.
- [Code Academy](https://www.codecademy.com/fr/learn/python)
- [newcoder.io](http://newcoder.io/). Des projets commentés, commencer par 'Data Visualization'
- [Google's Python Class](https://developers.google.com/edu/python/). Guido a travaillé chez eux.
  Pas [ce
  Guido](http://vignette2.wikia.nocookie.net/pixar/images/1/10/Guido.png/revision/latest?cb=20140314012724),
  [celui-là](https://en.wikipedia.org/wiki/Guido_van_Rossum#/media/File:Guido_van_Rossum_OSCON_2006.jpg)
- [Mooc Python](https://www.fun-mooc.fr/courses/inria/41001S03/session03/about#). Un mooc de
  l'INRIA, carré.
- [Code combat](https://codecombat.com/)

### Divers

- La chaîne YouTube [3blue1brown](https://www.youtube.com/c/3blue1brown) pour des vidéos de maths
  générales.
- La chaîne YouTube de [Freya Holmér](https://www.youtube.com/c/Acegikmo) plutôt orientée *game
  design*, mais avec d'excellentes vidéos de géométrie computationnelle.

## Licences

[![CC BY Licence
badge](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

Copyright © 2021 Loïc Grobol [\<loic.grobol@gmail.com\>](mailto:loic.grobol@gmail.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/legalcode>
