---
title: Apprentissage artificiel — M2 PluriTAL 2021
---

[comment]: <> "LTeX: language=fr"

- **Quoi** « Apprentissage Artificiel »
- **Où** Salle 219, bâtiment Paul Ricœur
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 22/09 au 17/11
  - Voir [le
    planning](http://www.tal.univ-paris3.fr/plurital/admin/Calendrier_M2_TAL_PX_2021_22.xlsx) pour
    les dates exactes
- **Contact** Loïc Grobol [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)

## Séances

Tous les supports sont sur [github](https://github.com/loicgrobol/apprentissage-artificiel), voir
[Utilisation en local](#utilisation-en-local) pour les utiliser sur votre machine comme des
notebooks. À défaut, ce sont des fichiers Markdown assez standards, qui devraient se visualiser
correctement sur la plupart des plateformes (mais ne seront pas dynamiques).

Les slides et les notebooks ci-dessous ont tous des liens Binder pour une utilisation interactive
sans rien installer. Les slides ont aussi des liens vers une version HTML statique utile si Binder
est indisponible.

### 2021-09-22 — Introduction et *crash course* Python

#### Slides

- [Slides 1](slides/lecture-01/lecture-01.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-01/lecture-01.md)
- [Slides 2](slides/lecture-02/lecture-02.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-02/lecture-02.md)
- [Slides 3](slides/lecture-03/lecture-03.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-03/lecture-03.md)
- [Slides 4](slides/lecture-04/lecture-04.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-04/lecture-04.md)
- [Slides 5](slides/lecture-05/lecture-05.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-05/lecture-05.md)

#### Exos

- [Power of Thor E01](https://www.codingame.com/ide/puzzle/power-of-thor-episode-1)
- [ASCII art](https://www.codingame.com/ide/puzzle/ascii-art)
- [The descent](https://www.codingame.com/ide/puzzle/the-descent)
- [Shadow of the knight E01](https://www.codingame.com/ide/puzzle/shadows-of-the-knight-episode-1)

#### Corrections

- [Exercices slides 1](slides/lecture-01/solutions-01.md) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-01/solutions-01.md)
- [Exercices slides 2](slides/lecture-02/solutions-02.md) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-02/solutions-02.md)
- Exercices Codingame : voir
  [Github](https://github.com/LoicGrobol//apprentissage-artificiel/tree/main/corrections)

### 2021-09-29 — Un peu de théorie et NumPy

- [Slides 6](slides/lecture-06/lecture-06.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-06/lecture-06.md)

#### Corrections

- [Exo sacs de mots](https://github.com/LoicGrobol//apprentissage-artificiel/tree/main/corrections/tfidf.py) dans sa version la plus sale possible.

### 2021-10-06 — Encore un peu de théorie, scikit-learn et les modèles de langues à n-grams

- [Slides 7](slides/lecture-07/lecture-07.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-07/lecture-07.md)
- Exercice : écrire deux scripts Python. Le premier doit apprendre à partir d'un corpus de textes un
  modèle de langue à n-grammes (avec n paramétrable) et le sauvegarder dans un fichier csv. L'autre
  doit lire le modèle précédent et l'utiliser pour générer une phrase. Tester avec [Le Ventre de
  Paris](data/zola_ventre-de-paris.txt), puis avec le corpus
  [CIDRE](https://www.ortolang.fr/market/corpora/cidre).

Lecture compagnon : [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 3 « *N-Gram language models* ».

Pour la fois prochaine : lire le chapitre 4 « *Naïve Bayes and Sentiment Classification* » (sauf
4.9) de [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/). Venir au cours
avec vos questions !

### 2021-10-13 — Modèles de langue à n-grammes (suite et fin) et *Naïve Bayes*

- [Slides 8](slides/lecture-08/lecture-08.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-08/lecture-08.md)
- [Slides 9](slides/lecture-09/lecture-09.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-09/lecture-09.md)

Lecture compagnon : [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 4 « *Naïve Bayes and Sentiment Classification* ».

Pour la fois prochaine :

- Faites de votre mieux pour les exercices à la fin du slide 9
- Relire le chapitre 4 « *Naïve Bayes and Sentiment Classification* » et lire le chapitre 5 « *Logistic Regression* », venir au cours avec vos questions !

### 2021-10-20 — Naïve Bayes (suite et fin) et régression logistique.

- [Correction slide 9](slides/lecture-09/correction.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-09/correction.md)
- [Slides 10](slides/lecture-10/lecture-10.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-10/lecture-10.md)

Lecture compagnon : [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 5 « *Logistic Regression* ».

Pour la fois prochaine :

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 6 « *Vector Semantics and Embeddings* ».

### 2021-10-27 — Régression logistique (suite et fin)

- [Slides 10](slides/lecture-10/lecture-10.slides.html) [![Launch in Binder
  badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-10/lecture-10.md)

Lecture compagnon : [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 5 « *Logistic Regression* ».

Pour la fois prochaine :

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
chapitre 6 « *Vector Semantics and Embeddings* ».


## Lire les slides en local

Les supports de ce cours sont écrits en Markdown, convertis en notebooks avec
[Jupytext](https://github.com/mwouts/jupytext). C'est entre autres une façon d'avoir un historique
git propre, malheureusement ça signifie que pour les ouvrir en local, il faut installer les
extensions adéquates. Le plus simple est le suivant

1. Récupérez le dossier du cours, soit en téléchargeant et décompressant
   [l'archive](https://github.com/LoicGrobol/apprentissage-artificiel/archive/refs/heads/main.zip)
   soit en le clonant avec git : `git clone
   https://github.com/LoicGrobol/apprentissage-artificiel.git` et placez-vous dans ce dossier.
2. Créez un environnement virtuel pour le cours (allez voir [le cours
   5](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-05/lecture-05.md)
   pour plus de détails sur ce que ça signifie)

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

   JupyterLab est aussi utilisable, mais la fonctionnalité slide n'y fonctionne pas pour l'instant.

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

Il y a beaucoup, beaucoup de ressources disponibles pour apprendre Python. Ce qui suit n'est qu'une sélection.

#### Livres

- How to think like a computer scientist, by Jeffrey Elkner, Allen B. Downey, and Chris Meyers. Vous
  pouvez l'acheter. Vous pouvez aussi le lire
  [ici](http://openbookproject.net/thinkcs/python/english3e/)
- Dive into Python, by Mark Pilgrim. [Ici](http://www.diveintopython3.net/) vous pouvez le lire ou
  télécharger le pdf.
- Learning Python, by Mark Lutz.
- Beginning Python, by Magnus Lie Hetland.
- Python Algorithms: Mastering Basic Algorithms in the Python Language, by Magnus Lie Hetland.
  Peut-être un peu costaud pour des débutants.
- Programmation Efficace. Les 128 Algorithmes Qu'Il Faut Avoir Compris et Codés en Python au Cours
  de sa Vie, by Christoph Dürr and Jill-Jênn Vie. Si le cours vous paraît trop facile. Le code
  Python est clair, les difficultés sont commentées. Les algos sont très costauds.

#### Web

Il vous est vivement conseillé d'utiliser un (ou plus) des sites et tutoriels ci-dessous.

- [Real Python](https://realpython.com), des cours et des tutoriels souvent de très bonne qualité et
  pour tous niveaux.
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

[![CC BY Licence badge](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

Copyright © 2021 Loïc Grobol [\<loic.grobol@gmail.com\>](mailto:loic.grobol@gmail.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à <https://creativecommons.org/licenses/by/4.0/legalcode>
