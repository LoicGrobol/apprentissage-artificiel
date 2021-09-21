# -*- coding: utf-8 -*-

"""
Implémentation de l'algo de calcul de la distance de Levenshtein
Voir http://www.xavierdupre.fr/app/mlstatpy/helpsphinx/c_dist/edit_distance.html
et https://fr.wikipedia.org/wiki/Distance_de_Levenshtein
Pour les implémentations : https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
"""

def distance(word_1, word_2):
    """
    calcule la distance d'édition entre les deux mots arguments
    """
    if len(word_1) < len(word_2):
        return distance(word_2, word_1)

    if word_1 == word_2:
        return 0
    elif len(word_1) == 0:
        return len(word_2)
    elif len(word_2) == 0:
        return len(word_1)
    else:
        matrix = {}
        word_1 = ' ' + word_1
        word_2 = ' ' + word_2
        W1 = len(word_1)
        W2 = len(word_2)
        for i in range(W1):
            matrix[i, 0] = i
        for j in range (W2):
            matrix[0, j] = j
        for i in range(1, W1):
            for j in range(1, W2):
                if word_1[i] == word_2[j]:
                    cost = 0
                else:
                    cost = 1
                matrix[i, j] = min(
                    matrix[i-1, j] + 1, # effacement
                    matrix[i, j-1] + 1, # insertion
                    matrix[i-1, j-1] + cost # substitution 
                    )
        return matrix[W1-1, W2-1]


def main():
    word_1, word_2 = ("roule", "raoul")
    print(f"{word_1}, {word_2}")
    print(distance(word_1, word_2))

if __name__ == "__main__":
    main()