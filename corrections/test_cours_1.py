# -*- coding: utf-8 -*-

"""
Tests pour exos du premier cours
"""
import triangles
from heures import heures, secondes
from carte import gagne_couleur

def test_triangles():
    assert triangles.caracteristiques(1, 1, 1) ==  ("equilatéral", 1)
    assert triangles.caracteristiques(1, 1, 2) == ("isocèle", 2)
    assert triangles.caracteristiques(1, 2, 1) == ("isocèle", 2)
    assert triangles.caracteristiques(2, 1, 1) == ("isocèle", 2)
    assert triangles.caracteristiques(2, 3, 1) == ("quelconque", 3)
    assert triangles.caracteristiques(2, 3, 6) == "pas un triangle"
    assert triangles.caracteristiques(6, 3, 2) == "pas un triangle"
    assert triangles.caracteristiques(2, 6, 3) == "pas un triangle"

def test_heures():
    assert(heures(0)) == "0:0:0"
    assert(heures(30)) == "0:0:30"
    assert(heures(60)) == "0:1:0"
    assert(heures(66)) == "0:1:6"
    assert(heures(3600)) == "1:0:0"
    assert(heures(86466)) == "24:1:6"
    assert(secondes('0:0:0')) == 0
    assert(secondes('6:6:6')) == 21966
    assert(secondes(heures(86466))) == 86466
    assert(heures(secondes('24:1:1'))) == "24:1:1"

def test_cartes():
    assert(gagne_couleur('S1', 'S2', 'S3', 'S4')) == 'S4'
    assert(gagne_couleur('S4', 'S3', 'S2', 'S1')) == 'S4'
    assert(gagne_couleur('S1', 'D2', 'C3', 'H4')) == 'S1'
    assert(gagne_couleur('S1', 'D2', 'S13', 'S10')) == 'S13'