# -*- coding: utf-8 -*-

"""
Tests pour exos du premier cours
"""
import austro
import levenshtein

def test_austro_words():
    csv_file = "../data/austronesian_swadesh.csv"
    assert austro.get_austro_words('Arab', 'Russian', []) == {'Arab':[], 'Russian':[]}
    assert austro.get_austro_words('Malay', 'Malagasy', ['new', 'old', 'good'], file=csv_file) == \
    {
        'Malay':['baharu', 'lama', 'bagus, baik'],
        'Malagasy':['vaovao', 'onta, hantitra', 'tsara']
    }
    assert austro.get_austro_words('Malay', 'Balinese', ['new', 'old', 'good'], file=csv_file) == \
    {
        'Malay':['baharu', 'lama', 'bagus, baik'],
        'Balinese':[]
    }

def test_same_prefix():
    assert austro.same_prefix('ako', 'ako') == True
    assert austro.same_prefix('ako', 'ika') == True
    assert austro.same_prefix('kamo', 'kayo') == True
    assert austro.same_prefix('unsa', 'nanu') == False

def test_distance():
    assert levenshtein.distance('roule', 'roule') == 0
    assert levenshtein.distance('roule', 'roules') == 1
    assert levenshtein.distance('roule', '') == 5
    assert levenshtein.distance('roule', 'raoul') == 2
    

