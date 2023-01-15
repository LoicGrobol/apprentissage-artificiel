# -*- coding: utf-8 -*-
from collections import Counter, namedtuple
import json
import re
import pathlib

import click
import numpy as np


def poor_mans_tokenizer_and_normalizer(s):
    return [
        w.lower()
        for w in re.split(r"\s|\W", s.strip())
        if w and all(c.isalpha() for c in w)
    ]


def get_counts(doc):
    return Counter(poor_mans_tokenizer_and_normalizer(doc))


def get_word_probs(bows, target):
    counts = np.ones((np.max(target) + 1, bows.shape[1]))
    for doc, c in zip(bows, target):
        counts[c] += doc
    total_per_class = np.sum(counts, axis=1, keepdims=True)
    return counts / total_per_class


Model = namedtuple(
    "Model", ["log_prior", "log_likelihood", "vocabulary", "target_names"]
)


def train_model(texts, target):
    target_names = sorted(set(target))
    target_voc = {c: i for i, c in enumerate(target_names)}
    target_array = np.fromiter((target_voc[t] for t in target), dtype=int)
    bows = [get_counts(doc) for doc in texts]
    vocab = sorted(set().union(*bows))
    w_to_i = {w: i for i, w in enumerate(vocab)}
    bow_array = np.zeros((len(bows), len(vocab)))
    for i, bag in enumerate(bows):
        for w, c in bag.items():
            bow_array[i, w_to_i[w]] = c

    log_prior = np.log(np.bincount(target_array) / target_array.size)
    log_likelihood = np.log(get_word_probs(bow_array, target_array))
    return Model(log_prior, log_likelihood, w_to_i, target_names)


# On pourrait faire une classe `Model` qui aurait ceci comme méthode
def save_model(model, model_dir):
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "vocabularies.json", "w") as out_stream:
        json.dump(
            {"words": model.vocabulary, "targets": model.target_names}, out_stream
        )
    np.save(model_dir / "log_prior.npy", model.log_prior)
    np.save(model_dir / "log_likelihood.npy", model.log_likelihood)


def load_model(model_dir):
    model_dir = pathlib.Path(model_dir)
    with open(model_dir / "vocabularies.json") as in_stream:
        vocabularies = json.load(in_stream)
    log_prior = np.load(model_dir / "log_prior.npy")
    log_likelihood = np.load(model_dir / "log_likelihood.npy")
    return Model(log_prior, log_likelihood, vocabularies["words"], vocabularies["targets"])


def predict_class(doc, model):
    bow_dict = get_counts(doc)
    bow = np.zeros(len(model.vocabulary))
    for w, c in bow_dict.items():
        word_idx = model.vocabulary.get(w)
        if word_idx is None:
            continue
        bow[word_idx] = c
    class_likelihoods = np.matmul(model.log_likelihood, bow) + model.log_prior
    return model.target_names[np.argmax(class_likelihoods)]


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "corpus_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "model_path", type=click.Path(file_okay=False, dir_okay=True, writable=True)
)
def train(corpus_path, model_path):
    corpus_path = pathlib.Path(corpus_path)
    classes_dirs = [d for d in corpus_path.glob("*") if d.is_dir()]
    texts = []
    targets = []
    for d in classes_dirs:
        for f in (f for f in d.glob("*") if f.is_file()):
            texts.append(f.read_text())
            targets.append(d.name)
    model = train_model(texts, targets)
    save_model(model, model_path)


@cli.command()
@click.argument(
    "model_path", type=click.Path(file_okay=False, dir_okay=True)
)
@click.argument(
    "texts", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1
)
def predict(model_path, texts):
    model = load_model(model_path)
    for doc in texts:
        print(predict_class(doc, model))


if __name__ == "__main__":
    cli()
