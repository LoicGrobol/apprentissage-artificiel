---
title: Apprentissage artificiel — M2 PluriTAL 2021
---

[comment]: <> "LTeX: language=fr"

## Infos pratiques

- **Quoi** « Apprentissage Artificiel »
- **Où** Salle 219, bâtiment Paul Ricœur
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 22/09 au 17/11
  - Voir [le
    planning](http://www.tal.univ-paris3.fr/plurital/admin/Calendrier_M2_TAL_PX_2021_22.xlsx) pour
    les dates exactes
- **Contact** Loïc Grobol [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)

## Séances

Tous les supports sont sur [github](https://github.com/loicgrobol/apprentissage-artificiel), les
liens vers les slides et les notebooks ci-dessous ont tous des liens Binder pour une utilisation
sans rien installer.

### 2021-09-22 — Introduction et *crash course* Python

Slides :

- [Slides 1](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-01/lecture-01.md)
- [Slides 2](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-02/lecture-02.md)
- [Slides 3](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-03/lecture-03.md)
- [Slides 4](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-04/lecture-04.md)
- [Slides 5](https://mybinder.org/v2/gh/loicgrobol/apprentissage-artificiel/main?urlpath=tree/slides/lecture-05/lecture-05.md)

Exos :

- [Power of Thor E01](https://www.codingame.com/ide/puzzle/power-of-thor-episode-1)
- [ASCII art](https://www.codingame.com/ide/puzzle/ascii-art)
- [The descent](https://www.codingame.com/ide/puzzle/the-descent)
- [Shadow of the knight E01](https://www.codingame.com/ide/puzzle/shadows-of-the-knight-episode-1)

## Outils

Vous aurez besoin d'un interpréteur Python et d'un éditeur de texte.

### Python & co

On travaillera avec Python 3.8 et supérieur.

Les supports de cours sont essentiellement sous forme de notebooks [Jupyter](http://jupyter.org/),
les diapos utilisant [RISE](https://github.com/damianavila/RISE). Pour utiliser les notebooks
(anciennement ipython notebook maintenant jupyter notebook) vous aurez besoin d'installer sur votre
machine de travail. Je vous incite également à utiliser le shell interactif `ipython` qui est une
version améliorée du shell `python` (ipython est inclus dans jupyter).

Deux options pour l'installation :

#### Installer uniquement les outils nécessaires avec pip

1. Installer Python 3, de préférence via le gestionnaire de paquets de votre système, sinon à partir
   de <https://www.python.org/downloads/>.
   Pour les distributions dérivées de Debian (y compris Ubuntu) vous aurez également besoin
   d'installer `pip`

      ```bash
      sudo apt install python3 python3-pip
      ```

2. Installer jupyter

      ```bash
      python3 -m pip install --user jupyter
      ```

#### Utiliser Anaconda

Ce n'est pas recommandé, mais si vous préférez, vous pouvez installer
[anaconda](https://www.continuum.io/downloads), qui gère non-seulement Python et les modules Python,
mais aussi beaucoup d'autres paquets et installera beaucoup de modules tiers dont on se servira pas

Nous verrons également dans le cours comment utiliser [virtualenv](https://virtualenv.pypa.io) pour
gérer des installations de Python isolées du système pour plus de confort.

## Ressources

Il y a beaucoup, beaucoup de ressources disponibles pour apprendre Python. Ce qui suit n'est qu'une sélection.

## Livres

- How to think like a computer scientist, by Jeffrey Elkner, Allen B. Downey, and Chris Meyers.
Vous pouvez l'acheter. Vous pouvez aussi le lire [ici](http://openbookproject.net/thinkcs/python/english3e/)
- Dive into Python, by Mark Pilgrim.
[Ici](http://www.diveintopython3.net/) vous pouvez le lire ou télécharger le pdf.
- Learning Python, by Mark Lutz.
- Beginning Python, by Magnus Lie Hetland.
- Python Algorithms: Mastering Basic Algorithms in the Python Language, by Magnus Lie Hetland.
Peut-être un peu costaud pour des débutants.
- Programmation Efficace. Les 128 Algorithmes Qu'Il Faut Avoir Compris et Codés en Python au Cours
  de sa Vie, by Christoph Dürr and Jill-Jênn Vie. Si le cours vous paraît trop facile. Le code
  Python est clair, les difficultés sont commentées. Les algos sont très costauds.

## Web

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

## Licences

[![CC BY Licence badge](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

Copyright © 2021 Loïc Grobol [\<loic.grobol@gmail.com\>](mailto:loic.grobol@gmail.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à <https://creativecommons.org/licenses/by/4.0/legalcode>
