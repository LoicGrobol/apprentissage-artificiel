# -*- coding: utf-8 -*-

"""
Python M2. Cours 1 : exercice sur les cartes à jouer
correction de Loïc Grobol (voir https://github.com/LoicGrobol/python-im-2)
"""

def gagne_couleur(carte1, carte2, carte3, carte4):
    """Affiche la carte qui remporte le pli en faisant attention aux couleurs :
        - la carte du premier joueur `carte1` donne la couleur attendue.
        - une carte qui n'est pas à la bonne couleur perd automatiquement.

    On ne gèrera pas certains cas incohérents comme une carte ou un pli invalide.
    """
    def gagne(carte1, carte2, couleur):  # on peut aussi définir une fonction dans une fonction
        if carte2[0] != couleur:
            return carte1
        elif int(carte1[1:]) < int(carte2[1:]):  # carte1 est forcément valide, pas besoin de vérifier
            return carte2
        else:
            return carte1
    couleur = carte1[0]
    gagnante = carte1
    for carte in [carte2, carte3, carte4]:
        gagnante = gagne(gagnante, carte, couleur)
    return gagnante