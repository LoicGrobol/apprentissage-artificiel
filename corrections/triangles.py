# -*- coding: utf-8 -*-

"""
Python M2. Cours 1 : exercice sur les triangles
correction de Loïc Grobol (voir https://github.com/LoicGrobol/python-im-2)
"""

def la_plus_grande(longueur1, longueur2, longueur3):
    """Renvoie la plus grande longueur."""
    return max(longueur1, longueur2, longueur3)

def est_equilateral(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est équilatéral."""
    return longueur1 == longueur2 and longueur2 == longueur3

def est_isocele(longueur1, longueur2, longueur3):
    """Renvoie si un triangle est isocele."""
    deux_egales = longueur1 == longueur2 or longueur1 == longueur3 or longueur2 == longueur3
    return deux_egales and not est_equilateral(longueur1, longueur2, longueur3)

def est_triangle(longueur1, longueur2, longueur3):
    """Renvoie si les longueurs données font bien un triangle."""
    maxi = la_plus_grande(longueur1, longueur2, longueur3)
    somme = longueur1 + longueur2 + longueur3
    return maxi <= (somme - maxi)  # la somme des deux côtés est au plus maxi

def caracteristiques(longueur1, longueur2, longueur3):
    """Affiche les caractéristiques d'un triangle.
    Les caractéristiques d'un triangle sont :
        - sa nature
        - la taille de son plus grand côté.

    On dira qu'un triangle est `quelconque` s'il n'est ni équilatéral ni isocèle.

    Affiche `pas un triangle` si les longueurs données ne font pas un triangle
    (la longueur du plus grand côté est supérieure à celle des deux autres).
    """
    if not est_triangle(longueur1, longueur2, longueur3):
        return "pas un triangle"
    else:
        maxi = la_plus_grande(longueur1, longueur2, longueur3)
        if est_equilateral(longueur1, longueur2, longueur3):
            return ("equilatéral", maxi)
        elif est_isocele(longueur1, longueur2, longueur3):
            return ("isocèle", maxi)
        else:
            return "quelconque", maxi
