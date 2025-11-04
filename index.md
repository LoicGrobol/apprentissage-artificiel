---
title: Apprentissage artificiel — M2 PluriTAL 2024
layout: default
---

<!-- LTeX: language=fr -->

## News

- **2025-09-23** Premier cours du semestre le 24/09/2025

## Infos pratiques

- **Quoi** « Apprentissage artificiel »
- **Où** Salle R06, BFC
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 24/09 au ??/11
  - Voir le planning pour les dates exactes
- **Contact** L. Grobol [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)
- **Évaluation** Un TP noté et un projet

## Liens utiles

- Prendre rendez-vous pour des *office hours* en visio :
  [mon calendrier](https://calendar.app.google/N9oW2c9BzhXsWrrv9)
- Lien Binder de secours :
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LoicGrobol/apprentissage-artificiel/main)

## Séances

Les liens dans chaque séance vous permettent de télécharger les fichiers `.ipynb` à utiliser (et
données additionnelles éventuelles). Attention: pour les utiliser en local il faudra installer les
packages du `requirements.txt` (dans un environnement virtuel). Si vous ne savez pas comment faire,
allez voir [« Utilisation en local »](#utilisation-en-local)

Les notebooks ci-dessous ont tous des liens Binder pour une utilisation interactive
sans rien installer.

### 2025-09-24 : Outils de travail

- {% notebook_badges slides/01-tools/python_crash_course.py.md %}
  [Crash course Python](slides/01-tools/python_crash_course.py.ipynb)
  - {% notebook_badges slides/01-tools/python_crash_course-solutions.py.md %}
    [Solutions](slides/01-tools/python_crash_course-solutions.py.ipynb)
- {% notebook_badges slides/01-tools/numpy-slides.py.md %}
  [Présentation Numpy](slides/01-tools/numpy-slides.py.ipynb)
- {% notebook_badges slides/01-tools/polars.py.md %}
  [Présentation Polars](slides/01-tools/polars.py.ipynb)

### 2025-09-31 : Intuitions et vocabulaire

- [Slides intro](slides/02-intro/intro.pdf)
- {% notebook_badges slides/02-scikit-learn/scikit-learn-slides.py.md %}
  [TP intro à scikit-learn](slides/02-scikit-learn/scikit-learn-slides.py.ipynb)
  - [`imdb_smol.tar.gz`](slides/02-scikit-learn/data/imdb_smol.tar.gz)

### 2025-10-08 : Évaluation de modèles, sur- et sous-apprentissage

- À lire sur les splits train/dev/test : Goot, Rob van der. 2021. ‘[We Need to Talk About
  Train-Dev-Test Splits](https://aclanthology.org/2021.emnlp-main.368)’. Proceedings of the 2021
  Conference on Empirical Methods in Natural Language Processing, November, 4485–94..


TP : full autonomie ! Préparez un notebook (avec du code **et** du texte) où vous utiliserez le
scikit-learn sur le jeu de données [20
newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) (récupérez
le pré-vectorisé en Tf⋅Idf avec
[`sklearn.datasets.fetch_20newsgroups_vectorized`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html))
pour étudier les capacités de quelques classifieurs.

- Présenter rapidement le jeu de données (origine, composition, quelques stats descriptives). Vous
  devriez probablement aller chercher dans la doc d'où proviennent les données et aller examiner
  leur contenu (quand on fait de la classification de textes, c'est souvent utile de regarder les
  textes en question!).
- Entraîner [des arbres de
  décision](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
  en faisant varier la profondeur pour étudier dans quelle mesure ils sous-apprennent quand la
  profondeur est trop faible et déterminer à partir de quand ils surapprennent. Penser à la
  validation croisée, penser à faire des courbes d'apprentissage, penser à visualiser les arbres,
  penser à regarder les matrices de confusion, chercher dans [les métriques de
  scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
  celles qui pourraient être intéressantes.
- Étudier l'influence de la taille du corpus d'apprentissage (mêmes indices)
- Si vous ne l'avez pas déjà fait, allez voir dans la doc à quoi sert le paramètre `remove` de
  [`fetch_20newsgroups_vectorized`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html),
  et reprenez vos expériences en le modifiant.
- Même jeu avec un
  [perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html),
  en étudiant l'influence du paramètre `max_iter` (commencer à 1)

Si vous voulez que j'y jette un œil, envoyez  à `lgrobol@parisnanterre.fr` avec en objet `[ML2025] TP
Overfit`.

### 2025-10-14 : Régression linéaire, régression logistique

- {% notebook_badges slides/04-logistic-regression/lr-slides.py.md %}
  [TP régression logistique](slides/04-logistic-regression/lr-slides.py.ipynb)
  - [Lexique VADER](slides/04-logistic-regression/data/vader_lexicon.txt)
  - [Jeu de données IMDB smol](slides/04-logistic-regression/data/imdb_smol.tar.gz)

### 2025-10-22 : Descente de gradient

- {% notebook_badges slides/04-logistic-regression/lr-slides.py.md %}
  [TP régression logistique](slides/04-logistic-regression/lr-slides.py.ipynb)
  - [Lexique VADER](slides/04-logistic-regression/data/vader_lexicon.txt)
  - [Jeu de données IMDB smol](slides/04-logistic-regression/data/imdb_smol.tar.gz)

### 2025-11-05 : Apprentissage non-supervisé

- {% notebook_badges slides/05-unsupervised/unsupervised.py.md %}
  [TP apprentissage non-supervisé](slides/05-unsupervised/unsupervised.py.ipynb)

## Évaluations

### TP 20ng

À rendre au plus tard le 2025-12-19T23:59:59
  ([UTC-12](https://fr.wikipedia.org/wiki/Partout_sur_Terre)): un notebook qui récupère les données,
  entraîne un (ou plusieurs) modèle et l'évalue.

- Vous serez évalué⋅es sur les performances du modèle, le temps d'exécution du notebook, la qualité
  du code et la qualité de vos explications.
- Le notebook doit s'exécuter sur une machine standard, avec Python 3.12, 3.13 ou 3.14 et sans
  configuration non-documentéee.
- Si vous utilisez des packages autres que ceux utilisés dans le cours **documentez-les** au début
  du notebook.
- Si votre notebook ne fonctionne pas sans que j'ai à le modifier, ce sera pénalisé.
- Les résultats doivent être clairement affichés à la fin. Cet affichage doit être *généré* (pas
  écrit à la main).
- Les résultats doivent être exactement les mêmes à chaque exécution du notebook (après un
  redémarrage du kernel). Pensez aux *random seed*.
- Si vous mettez du texte explicatif, des titres, etc., c'est mieux. **A minima le notebook doit
  contenir vos noms, prénoms et établissement principal**.
- Les données doivent être celles venant de scikit-learn.
- Vous devez utiliser le split train/test standard.
- Vous pouvez utiliser une vectorisation et des prétraitements différents de ceux par défaut.
- Les données **doivent** être chargées avec `remove=("headers", "footers", "quotes")` (voir [la
  doc](https://scikit-learn.org/1.5/datasets/real_world.html#filtering-text-for-more-realistic-training)
  pour savoir pourquoi.)
- Tous les coups sont permis : vous avez droit à tout sauf aux réseaux de neurones (à part
éventuellement un perceptron).
- Les choix d'hyperparamètres différents de ceux par défauts doivent être justifiés (dans des
cellules de textes) de façon aussi convaincante que possible). Si vous utilisez des algorithmes de
recherches (type grid search), ils doivent faire partie du notebook (et donc compter dans le temps
d'exécution).
- Rendus par mail à `lgrobol@parisnanterre.fr` avec en objet `[ML2024] TP 20ng` et vos noms, prénoms
et établissement dans le corps du mail.
- **Si l'objet est différent, je ne verrai pas votre rendu**.
- J'accuserai réception sous 48h dans la mesure du possible, relancez-moi si ce n'est pas le cas.

## Utilisation en local

### Environnements virtuels et packages

Je cite le [Crash course Python](slides/01-tools/python_crash_course.py.ipynb):

- Les environnements virtuels sont des installations isolées de Python. Ils vous permettent d'avoir
  des versions indépendantes de Python et des packages que vous installez
  - Gérez vos environnements et vos packages avec [uv](https://docs.astral.sh/uv/). Installez-le,
    lisez la doc.
  - Pour créer un environnement virtuel : `uv venv /chemin/vers/…`
  - La convention, c'est `uv venv .venv`, ce qui créée un dossier (caché par défaut sous Linux et Mac
    OS car son nom commence par `.`) : `.venv` dans le dossier courant (habituellement le dossier
    principal de votre projet). Donc faites ça.
  - Il est **obligatoire** de travailler dans un environnement virtuel. L'idéal est d'en avoir un
    par cours, un par projet, etc.
    - uv est assez précautionneux avec l'espace disque, il y a donc assez peu de désavantage à avoir
      beaucoup d'environnements virtuels.
  - Un environnement virtuel doit être **activé** avant de s'en servir. Concrètement ça remplace la
    commande `python` de votre système par celle de l'environnement.
    - Dans Bash par exemple, ça se fait avec `source .venv/bin/activate` (en remplaçant par le
      chemin de l'environnement s'il est différent)
    - `deactivate` pour le désactiver et rétablir votre commande `python`. À faire avant d'en
      activer un autre.
- On installe des packages avec `uv pip` ou `python -m pip` (mais plutôt `uv pip`, et jamais juste
  `pip`).
  - `uv pip install numpy` pour installer Numpy.
  - Si vous avez un fichier avec un nom de package par ligne (par exemple le
    [`requirements.txt`](https://github.com/LoicGrobol/apprentissage-artificiel/blob/main/requirements.txt)
    du cours) : `uv pip install -U -r requirements.txt`
  - Le flag `-U` ou `--upgrade` sert à mettre à jour les packages si possible : `uv pip install -U
    numpy` etc.
- Je répète : on installe uniquement dans un environnement virtuel, on garde ses environnements bien
  séparés (un par cours, pas un pour tout le M2).
  - Dans un projet, on note dans un `requirements.txt` (ou `.lst`) les packages dont le projet a
    besoin pour fonctionner.
  - Les environnements doivent être légers : ça ne doit pas être un problème de les effacer, de les
    recréer… Si vous ne savez pas recréer un environnement que vous auriez perdu, c'est qu'il y a un
    problème dans votre façon de les gérer.
- Si vous voulez en savoir plus, **et je recommande très fortement de vouloir en savoir plus, c'est
  vital de connaître ses outils de travail**, il faut : *lire les documentations de **tous** les
  outils et **toutes** les commandes que vous utilisez*.

Maintenant à vous de jouer :

- Installez uv
- Créez un dossier pour ce cours
- Dans ce dossier, créez un environnement virtuel nommé `.venv`
- Activez-le
- Téléchargez le
  [`requirements.txt`](https://github.com/LoicGrobol/apprentissage-artificiel/blob/main/requirements.txt)
  et installez les packages qu'il liste

### Notebooks Jupyter

Si vous avez une installation propre (par exemple en suivant les étapes précédentes), vous pouvez
facilement ouvrir les notebooks du cours :

- Téléchargez le notebook du [Crash course Python](slides/01-tools/python_crash_course.py.ipynb) et
  mettez-le dans le dossier que vous utilisez pour ce cours.
- Dans un terminal (avec votre environnement virtuel activé) lancez jupyter avec `jupyter notebook
  python_crash_course.py.ipynb`.
- Votre navigateur devrait s'ouvrir directement sur le notebook. Si ça ne marche pas, le terminal
  vous donne dans tous les cas un lien à suivre.

Alternativement, des IDE comme vscode permettent d'ouvrir directement les fichiers ipynb. Pensez à
lui préciser que le kernel a utiliser est celui de votre environnement virtuel s'il ne le trouve pas
tout seul.

### Utilisation avancée

Vous pouvez aussi (mais je ne le recommande pas forcément car ce sera plus compliqué pour vous de le
maintenir à jour) cloner [le dépôt du
cours](https://github.com/loicgrobol/apprentissage-artificiel). Tous les supports y sont, sous forme
de fichiers Markdown assez standards, qui devraient se visualiser correctement sur la plupart des
plateformes. Pour les utiliser comme des notebooks, il vous faudra utiliser l'extension
[Jupytext](https://github.com/mwouts/jupytext) (qui est dans le `requirements.txt`). C'est entre
autres une façon d'avoir un historique git propre.

## Ressources

### Apprentissage artificiel

La référence pour le TAL :

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et
  James H. Martin est **indispensable**. Il parle de bien plus de chose que simplement de
  l'apprentissage artificiel, mais sur le plan théorique il contient tout ce dont on parlera
  concernant l'apprentissage pour le TAL. Il est disponible gratuitement et mis à jour tous les ans,
  donc n'hésitez pas à le consulter très fréquemment.
  
 Les suivants sont des textbooks avec une approche mathématique plus complète et détaillée, c'est
 vers eux qu'il faut se tourner pour répondre aux questions profondes. Ils sont un peu cher alors si
 vous voulez les utiliser, commencez par me demander et je vous prêterai les miens.

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732)
  Christopher M. Bishop (2006), le textbook classique.
- [Machine Learning: A Probabilistic
  Perspective](https://mitpress.mit.edu/9780262018029/machine-learning/) de Kevin P. Murphy, (2012)
  on peut difficilement faire plus complet.
- [*Apprentissage artificiel - Concepts et
  algorithmes*](https://www.eyrolles.com/Informatique/Livre/apprentissage-artificiel-9782416001048/)
  d'Antoine Cornuéjols et Laurent Miclet. (En français!)

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
