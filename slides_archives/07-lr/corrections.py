import math
import re
from collections import defaultdict
import pathlib
import random

import numpy as np

def crude_tokenizer_and_normalizer(s):
    tokenizer_re = re.compile(
        r"""
        (?:                   # Dans ce groupe, on détecte les mots
            \b\w+?\b          # Un mot c'est des caractères du groupe \w, entre deux frontières de mot
            (?:               # Éventuellement suivi de
                '             # Une apostrophe
            |
                (?:-\w+?\b)*  # Ou d'autres mots, séparés par des traits d'union
            )?
        )
        |\S        # Si on a pas détecté de mot, on veut bien attraper un truc ici sera forcément une ponctuation
        """,
        re.VERBOSE,
    )
    return tokenizer_re.findall(s.lower())


def read_vader(vader_path):
    res = dict()
    with open(vader_path) as in_stream:
        for row in in_stream:
            word, polarity, *_ = row.lstrip().split("\t", maxsplit=2)
            res[word] = float(polarity)
    return res


def featurize(text, lexicon):
    words = crude_tokenizer_and_normalizer(text)
    features = np.empty(2)
    # Le max permet de remonter les polarités négatives à 0
    features[0] = sum(max(lexicon.get(w, 0), 0) for w in words)/len(words)
    features[1] = sum(max(-lexicon.get(w, 0), 0) for w in words)/len(words)
    return features


def featurize_dir(corpus_root, lexicon):
    corpus_root = pathlib.Path(corpus_root)
    res = defaultdict(list)
    for clss in corpus_root.iterdir():
        # On peut aussi utiliser une compréhension de liste et avoir un dict pas default
        for doc in clss.iterdir():
            # `stem` et `read_text` c'est de la magie de `pathlib`, check it out
            res[clss.stem].append(featurize(doc.read_text(), lexicon))
    return res


def affine_combination(x, w, b):
    return np.inner(w, x) + b


def hardcoded_classifier(x):
    return affine_combination(x, np.array([0.6, -0.4]), -0.01) > 0.0


def classifier_accuracy(w, b, featurized_corpus):
    correct_pos = sum(1 for doc in featurized_corpus["pos"] if affine_combination(doc, w, b) > 0.0)
    correct_neg = sum(1 for doc in featurized_corpus["neg"] if affine_combination(doc, w, b) <= 0.0)
    return (correct_pos+correct_neg)/(len(featurized_corpus["pos"])+len(featurized_corpus["neg"]))


def logistic(z):
    return 1/(1+np.exp(-z))


def logistic_negative_log_likelihood(x, w, b, y):
    g_x = logistic(affine_combination(x, w, b))
    if y == 1:
        correct_likelihood = g_x
    else:
        correct_likelihood = 1-g_x
    loss = -np.log(correct_likelihood)
    return loss


def loss_on_imdb(w, b, featurized_corpus):
    loss_on_pos = math.fsum(
        logistic_negative_log_likelihood(doc_features, w, b, 1).astype(float)
        for doc_features in featurized_corpus["pos"]
    )
    loss_on_neg = math.fsum(
        logistic_negative_log_likelihood(doc_features, w, b, 0).astype(float)
        for doc_features in featurized_corpus["neg"]
    )
    return np.array([loss_on_pos + loss_on_neg])


def grad_L(x, w, b, y):
    g_x = logistic(np.inner(w, x) + b)
    grad_w = (g_x - y)*x
    grad_b = g_x - y
    return np.append(grad_w, grad_b)


def descent(featurized_corpus, theta_0, learning_rate, n_steps):
    train_set = [
        *((doc, 1) for doc in featurized_corpus["pos"]),
        *((doc, 0) for doc in featurized_corpus["neg"])
    ]
    theta = theta_0
    w = theta[:-1]
    b = theta[-1]
    
    for i in range(n_steps):
        # On mélange le corpus pour s'assurer de ne pas avoir d'abord tous
        # les positifs puis tous les négatifs
        random.shuffle(train_set)
        for j, (x, y) in enumerate(train_set):
            grad = grad_L(x, w, b, y)
            steepest_direction = -grad
            theta += learning_rate*steepest_direction
            w = theta[:-1]
            b = theta[-1]
    return (w, b)


def descent_with_logging(featurized_corpus, theta_0, learning_rate, n_steps):
    train_set = [
        *((doc, 1) for doc in featurized_corpus["pos"]),
        *((doc, 0) for doc in featurized_corpus["neg"])
    ]
    theta = theta_0
    theta_history = [theta_0.tolist()]
    w = theta[:-1]
    b = theta[-1]
    print("Epoch\tLoss\tAccuracy\tw\tb")
    print(f"Initial\t{loss_on_imdb(w, b, featurized_corpus).item()}\t{classifier_accuracy(w, b, featurized_corpus)}\t{w}\t{b}")
    
    for i in range(n_steps):
        # On mélange le corpus pour s'assurer de ne pas avoir d'abord tous
        # les positifs puis tous les négatifs
        random.shuffle(train_set)
        for j, (x, y) in enumerate(train_set):
            grad = grad_L(x, w, b, y)
            steepest_direction = -grad
            # Purement pour l'affichage
            loss = logistic_negative_log_likelihood(x, w, b, y)
            #print(f"step {i*len(train_set)+j} doc={x}\tw={w}\tb={b}\tloss={loss}\tgrad={grad}")
            theta += learning_rate*steepest_direction
            w = theta[:-1]
            b = theta[-1]
        theta_history.append(theta.tolist())
        epoch_train_loss = loss_on_imdb(w, b, featurized_corpus).item()
        epoch_train_accuracy = classifier_accuracy(w, b, featurized_corpus)
        print(f"{i}\t{epoch_train_loss}\t{epoch_train_accuracy}\t{w}\t{b}")
    return (theta[:-1], theta[-1]), theta_history
