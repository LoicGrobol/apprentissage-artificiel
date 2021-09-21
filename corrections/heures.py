# -*- coding: utf-8 -*-

"""
Python M2. Cours 1 : exercice sur les heures et secondes
correction de Loïc Grobol (voir https://github.com/LoicGrobol/python-im-2)
"""

def heures(secondes):
    """Prend un nombre de secondes (entier) et le convertit en heures, minutes
    et secondes sous le format `H:M:S` où `H` est le nombre d'heures,
    `M` le nombre de minutes et `S` le nombre de secondes.

    On suppose que secondes est positif ou nul (secondes >= 0).
    """
    H = secondes // 3600
    M = (secondes % 3600) // 60
    S = secondes % 60
    return f"{H}:{M}:{S}"

def secondes(heure):
    """Prend une heure au format `H:M:S` et renvoie le nombre de secondes
    correspondantes (entier).

    On suppose que l'heure est bien formattée. On aura toujours un nombre
    d'heures valide, un nombre de minutes valide et un nombre de secondes valide.
    """
    H, M, S = heure.split(":")
    return (3600 * int(H)) + (60 * int(M)) + int(S)
