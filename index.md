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
- **Évaluation** Un TP noté en temps limité (date à déterminer) et un projet

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

## Utilisation en local

### Environnements virtuels et packages

Je cite le [Crash course Python](slides/01-tools/python_crash_course.py.ipynb):

- Les environnements virtuels sont des installations isolées de Python. Ils vous permettent d'avoir
  des versions indépendantes de Python et des packages que vous installez
  - Gérez vos environnements et vos packages avec [uv](https://docs.astral.sh/uv/). Installez-le,
    lisez la doc.
  - Pour créer un environnement virtuel : `uv venv /chemin/vers/…`
  - La convention, c'est `uv venv .venv`, ce qui créée un dossier (caché par défaut sous Linux et Mac
    OS car son nom commence par  point) `.venv` dans le dossier courant (habituellement le dossier
    principal de votre projet). Donc faites ça.
  - Il est **obligatoire** de travailler dans un environnement virtuel. L'idéal est d'en avoir un
    par cours, un par projet, etc.
  - Un environnement virtuel doit être **activé** avant de s'en servir. Concrètement ça remplace la
    commande `python` de votre système par celle de l'environnement.
    - Sous Bash, ça se fait avec `source .venv/bin/activate` (en remplaçant par le chemin de
      l'environnement s'il est différent)
    - `deactivate` pour le désactiver et rétablir votre commande `python`. À faire avant d'en
      activer un autre.
- On installe des packages avec `uv pip` ou `python -m pip` (mais plutôt `uv pip`, et jamais juste
  `pip`).
  - `uv pip install numpy` pour installer Numpy.
  - Si vous avez un fichier avec un nom de package par ligne (par exemple le
    [`requirements.txt`](https://github.com/LoicGrobol/apprentissage-artificiel/blob/main/requirements.txt)
    du cours) : `uv pip install -U -r requirements.txt`
  - Le flag `-U` ou `--upgrade` sert à mettre à jour les packages si possible : `uv pip install -U numpy` etc.
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
  mettez-le dans le dossier du cours.
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
[Jupytext](https://github.com/mwouts/jupytext). C'est entre autres une façon d'avoir un historique
git propre, malheureusement ça signifie que pour les ouvrir en local, il faut installer des trucs.
Le plus simple est le suivant

1. Récupérez le dossier du cours, soit en téléchargeant et décompressant
   [l'archive](https://github.com/LoicGrobol/apprentissage-artificiel/archive/refs/heads/main.zip)
   soit en le clonant avec git : `git clone
   https://github.com/LoicGrobol/apprentissage-artificiel.git` et placez-vous dans ce dossier.
2. Créez un environnement virtuel pour le cours (par exemple ici avec [virtualenv](https://virtualenv.pypa.io)) 

   ```console
   uv venv .venv
   source .venv/bin/activate
   ```

3. Installez les dépendances

   ```console
   uv pip install -U -r requirements.txt
   ```

4. Lancez Jupyter

   ```console
   jupyter notebook
   ```

   JupyterLab est aussi utilisable.

## Ressources

### Apprentissage artificiel

La référence pour le TAL :

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et
  James H. Martin est **indispensable**. Il parle de bien plus de chose que simplement de
  l'apprentissage artificiel, mais sur le plan théorique il contient tout ce dont on parlera
  concernant l'apprentissage pour le TAL. Il est disponible gratuitement et mis à jour tous les ans, donc n'hésitez pas à le
  consulter très fréquemment.
  
 Les suivants sont des textbook avec une approche mathématique plus complète et détaillée, c'est vers eux qu'il faut se tourner pour répondre aux questions profondes. Ils sont un peu cher alors si vous voulez les utiliser, commencez par me demander et je vous prêterai les miens.
 
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) Christopher M. Bishop (2006), le textbook classique.
- [Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/9780262018029/machine-learning/) de Kevin P. Murphy, (2012) on peut difficilement faire plus complet.
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
