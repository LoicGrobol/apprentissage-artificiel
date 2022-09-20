---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
Cours 13 : Réseaux de neurones pour le traitement de séquences
==============================================================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

2021-11-15
<!-- #endregion -->

```python
from IPython.display import display, Markdown
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

## POS tagging

## Récupérer les données avec 🤗 datasets


🤗 datasets ?


[🤗 datasets](https://huggingface.co/docs/datasets).

```python
%pip install -U conllu datasets
```

```python
from datasets import load_dataset
dataset = load_dataset(
   "universal_dependencies", "fr_sequoia"
)
```

```python
dataset
```

```python
train_dataset = dataset["train"]
print(train_dataset.info.description)
```

```python
train_dataset[5]
```

```python
train_dataset[5]["tokens"]
```

```python
train_dataset.features
```

```python
train_dataset.features["upos"]
```

```python
train_dataset.features["upos"].feature.names[0]
```

```python
upos_names = train_dataset.features["upos"].feature.names
```

```python
[upos_names[i] for i in train_dataset[5]["upos"]]
```

Et une fonction pour faire la traduction

```python
def get_pos_names(pos_indices):
    return [upos_names[i] for i in pos_indices]

get_pos_names(train_dataset[5]["upos"])
```

Il nous reste un truc à faire : construire un dictionnaire de mots pour passer des tokens du dataset à des nombres. On connaît la chanson : d'abord on récupère le vocabulaire.

```python
from collections import Counter
word_counts = Counter(t.lower() for row in train_dataset for t in row["tokens"])
word_counts.most_common(16)
```

On filtre les hapax et on trie par ordre alphabétique pour que notre vocabulaire ne change pas d'une exécution sur l'autre

```python
idx_to_token = sorted([t for t, c in word_counts.items() if c > 1])
idx_to_token[-8:]
```

Combien de mots ça nous fait ?

```python
len(idx_to_token)
```

Finalement on construit un dictionnaire pour avoir la transformation inverse

```python
token_to_idx = {t: i for i, t in enumerate(idx_to_token)}
token_to_idx["demain"]
```

Une fonction pour lire le dataset et récupérer les tokens et les POS comme tenseurs entiers. Le seul souci ici c'est qu'on a des mots inconnus et qu'il faudra leur attribuer un indice aussi : on va leur donner tous `len(idx_to_tokens)`.

```python
import torch

def encode(tokens):
    words_idx = torch.tensor(
        [
            token_to_idx.get(t.lower(), len(token_to_idx))
            for t in tokens
        ],
        dtype=torch.long,
    )
    return words_idx

def vectorize(row):
    words_idx = encode(row["tokens"])
    pos = torch.tensor(row["upos"], dtype=torch.long)
    return (words_idx, pos)

vectorize(train_dataset[5])
```

## Avec un FFNN


Premier test : on va juste faire un classifieur neuronal tout simple, ça nous permettra de voir les bases. On fera plus fancy après.


On prend la structure du classifieur neuronal de la dernière fois avec un petit changement : au lieu de passer des vecteurs de features qu'on déterminait nous même, on va lui passer les entiers qui représentent les mots et les passer par une couche de plongement `Embedding` qui nous donnera des vecteurs de mots statiques qui joueront le rôle de features.

Sinon c'est la même sauce : une couche cachée dense, une couche de sortie, un sofmax et une log-vraisemblance négative comme loss.

```python
from typing import Sequence
import torch.nn

class SimpleClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int
    ):
        # Une idiosyncrasie de torch, pour qu'iel puisse faire sa magie
        super().__init__()
        # On ajoute un mot supplémentaire au vocabulaire : on s'en servira pour les mots inconnus
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        self.hidden = torch.nn.Linear(embeddings_dim, hidden_size)
        self.hidden_activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, n_classes)
        # Comme on va calculer la log-vraisemblance, c'est le log-softmax qui nous intéresse
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt):
        emb = self.embeddings(inpt)
        hid = self.hidden_activation(self.hidden(emb))
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        # Pas de calcul de gradient ici : c'est juste pour les prédictions
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

    
source, target = vectorize(train_dataset[5])
display(source)
display(target)
display(get_pos_names(target))
simple_classifier = SimpleClassifier(len(idx_to_token), 128, 512, len(upos_names))
with torch.no_grad():
    output = simple_classifier(source)
display(output)
output_predictions = output.argmax(dim=-1)
display(output_predictions)
display(get_pos_names(output_predictions))

simple_classifier.predict(["Le", "petit", "chat", "est", "content"])
```

Évidemment c'est n'importe quoi : on a pas encore entraîné !


Pour entraîner c'est comme précédemment, descente de gradient yada yada.

Cette fois-ci au lieu de SGD on va utiliser Adam qui a tendance à mieux marcher en général.

```python
import random
from typing import Sequence, Tuple
import torch.optim

# Pour s'assurer que les résultats seront les mêmes à chaque run du notebook
torch.use_deterministic_algorithms(True)

def train_network(
    model: torch.nn.Module,
    train_set: Sequence[Tuple[torch.tensor, torch.Tensor]],
    dev_set: Sequence[Tuple[torch.tensor, torch.Tensor]],
    epochs: int
):
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Epoch\ttrain loss\tdev accuracy")
    for epoch_n in range(epochs):
    
        epoch_loss = 0.0
        epoch_length = 0
        for source, target in train_set:
            optim.zero_grad()
            out = model(source)
            loss = torch.nn.functional.nll_loss(out, target)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_length += source.shape[0]

        dev_correct = 0
        dev_total = 0
        for source, target in dev_set:
            # Ici on ne se sert pas du gradient, on évite donc de le calculer
            with torch.no_grad():
                out_prediction = model(source).argmax(dim=-1)
                dev_correct += out_prediction.eq(target).sum()
                dev_total += source.shape[0]
        print(f"{epoch_n}\t{epoch_loss/epoch_length}\t{dev_correct/dev_total:.2%}")

torch.manual_seed(0)
random.seed(0)
trained_classifier = SimpleClassifier(len(idx_to_token), 128, 512, len(upos_names))
train_network(
    trained_classifier,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

```python
trained_classifier.predict(["Le", "petit", "chat","est", "content", "."])
```

```python
trained_classifier.predict(["Le", "ministre", "prend", "la", "fuite"])
```

```python
trained_classifier.predict(["L'", "état", "proto-fasciste", "applique", "une", "politique", "délétère", "."])
```

Problèmes :

- Pas d'accès au contexte : en fait on apprend un dictionnaire !
- Sans accès au contexte, le réseau a peu d'infos pour décider et donc a tendance à tomber dans l'heuristique de la classe majoritaire.
- Surtout pour les mots inconnus


Un peu mieux : on va donner accès non seulement au mots mais aussi aux contexte gauches et droits

```python
class ContextClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        # La couche cachée va prendre des trigrammes en entrée, du coup elle doit être
        # plus grande
        self.hidden = torch.nn.Linear(3*embeddings_dim, hidden_size)
        self.hidden_activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt):
        emb = self.embeddings(inpt)
        # On va ajouter des faux mots avant et après comme remplissage pour les trigrammes
        emb = torch.cat(
            [
                torch.zeros(1, emb.shape[1]),
                emb,
                torch.zeros(1, emb.shape[1]),
            ],
            dim=0,
        )
        # La représentation d'un token ça va être la concaténation de son embedding et de ceux
        # des tokens d'avant et d'après
        hid_input = torch.cat([emb[:-2], emb[1:-1], emb[2:]], dim=-1)
        hid = self.hidden_activation(self.hidden(hid_input))
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

context_classifier = ContextClassifier(len(idx_to_token), 128, 512, len(upos_names))
context_classifier.predict(["Le", "petit", "chat", "est", "content"])
```

On l'entraîne

```python
torch.manual_seed(0)
random.seed(0)
context_classifier = ContextClassifier(len(idx_to_token), 128, 512, len(upos_names))
train_network(
    context_classifier,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

```python
context_classifier.predict(["Le", "petit", "chat","est", "content", "."])
```

```python
context_classifier.predict("Je reconnais l' existence du kiwi .".split())
```

C'est un peu mieux mais

- La prise en compte du contexte est pas encore parfaite
- Il galère toujours avec les mots hors vocabulaire


En pratique on peut faire beaucoup même avec des modèles de ce type en les aidant plus

- Ajouter plus de données
- Mettre plus de couches ou des couches plus larges
- Mettre du dropout
- Éventuellement donner plus de contexte
- Pré-entraîner les embeddings
- …

Comme d'hab *Natural Language Processing (almost) from Scratch* (Collobert et al., 2011) a plein de bons exemples.


Cependant, on reste sur un truc frustrant : le contexte pris en compte est limité, on aimerait bien plutôt pouvoir prendre en compte toute la phrase.


On va voir une famille de réseaux de neurones qui permettent de modéliser ça directement.

## Réseaux de neurones récurrents


[![](http://dprogrammer.org/wp-content/uploads/2019/04/RNN_Core2-768x491.png)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


[![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


Illustrations :

- [RNN, LSTM & GRU](http://dprogrammer.org/rnn-lstm-gru)
- [Les RNN, les LSTM, les GRU et ELMo](https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/)
- [Illustrated Guide to RNN, LSTM, and Transformers](http://sungsoo.github.io/2021/06/29/illustrated-guide.html) (vidéo)


Un tagger avec une couche cachée récurrente

```python
class RNNTagger(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        self.hidden = torch.nn.RNN(embeddings_dim, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt):
        emb = self.embeddings(inpt)
        # La couche `RNN` attends des entrées de dimension
        # taille de batch×longueur de séquence×features
        emb = emb.view(1, emb.shape[0], emb.shape[1])
        # La sortie est un couple avec les sorties pour chacun des éléments
        # de la séquence, plus l'état récurrent final
        hid, _ = self.hidden(emb)
        # `hid` est aussi de dimension taille de batch×longueur de séquence×features
        # Ici la taille de batch est 1, on l'enlève
        hid = hid.view(hid.shape[1], hid.shape[2])
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

recurrent_tagger = RNNTagger(len(idx_to_token), 128, 512, len(upos_names))
recurrent_tagger.predict(["Le", "petit", "chat", "est", "content"])
```

```python
torch.manual_seed(0)
random.seed(0)
recurrent_tagger = RNNTagger(len(idx_to_token), 128, 256, len(upos_names))
train_network(
    recurrent_tagger,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

```python
recurrent_tagger.predict(["Le", "chat", "est", "content"])
```

```python
recurrent_tagger.predict(["Le", "ministre", "prend", "la", "fuite"])
```

```python
class BiRNNTagger(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        self.hidden = torch.nn.RNN(
            embeddings_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # C'est un RNN bidirectionnel, donc il sort deux fois plus de features
        self.output = torch.nn.Linear(2*hidden_size, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt):
        emb = self.embeddings(inpt)
        emb = emb.view(1, emb.shape[0], emb.shape[1])
        hid, _ = self.hidden(emb)
        hid = hid.view(hid.shape[1], hid.shape[2])
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

birecurrent_tagger = BiRNNTagger(len(idx_to_token), 128, 512, len(upos_names))
birecurrent_tagger.predict(["Le", "petit", "chat", "est", "content"])
```

```python
torch.manual_seed(0)
random.seed(0)
birecurrent_tagger = BiRNNTagger(len(idx_to_token), 128, 256, len(upos_names))
train_network(
    birecurrent_tagger,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

```python
birecurrent_tagger.predict(["Le", "chat", "est", "content"])
```

```python
birecurrent_tagger.predict(["Le", "ministre", "prend", "la", "fuite"])
```

## LSTM


Une bonne source : [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


[![Schéma traditionnel d'un LSTM](https://upload.wikimedia.org/wikipedia/commons/6/63/Long_Short-Term_Memory.svg)](https://en.wikipedia.org/wiki/File:Long_Short-Term_Memory.svg)


[![](https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/RNN-LSTM-GRU-ELMO/LSTM%20architechture.png)](https://lbourdois.github.io/blog/nlp/RNN-LSTM-GRU-ELMO/)


[![](http://dprogrammer.org/wp-content/uploads/2019/04/LSTM-Core-768x466.png)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


$$\begin{align}
f_t &= \sigma_g(W_{f} x_t + U_{f} h_{t-1} + b_f) \\
i_t &= \sigma_g(W_{i} x_t + U_{i} h_{t-1} + b_i) \\
o_t &= \sigma_g(W_{o} x_t + U_{o} h_{t-1} + b_o) \\
\tilde{c}_t &= \sigma_c(W_{c} x_t + U_{c} h_{t-1} + b_c) \\
c_t &= f_t \circ c_{t-1} + i_t \circ \tilde{c}_t \\
h_t &= o_t \circ \sigma_h(c_t)
\end{align}$$

```python
class BiLSTMTagger(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        self.hidden = torch.nn.LSTM(
            embeddings_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.output = torch.nn.Linear(2*hidden_size, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        emb = self.embeddings(inpt)
        emb = emb.view(1, emb.shape[0], emb.shape[1])
        hid, _ = self.hidden(emb)
        hid = hid.view(hid.shape[1], hid.shape[2])
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

bilstm_tagger = BiLSTMTagger(len(idx_to_token), 128, 256, len(upos_names))
bilstm_tagger.predict(["Le", "petit", "chat", "est", "content"])
```

```python
torch.manual_seed(0)
random.seed(0)
bilstm_tagger = BiLSTMTagger(len(idx_to_token), 128, 256, len(upos_names))
train_network(
    bilstm_tagger,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

```python
bilstm_tagger.predict(["Le", "chat", "est", "content"])
```

```python
bilstm_tagger.predict(["Le", "ministre", "prend", "la", "fuite"])
```

## Mécanisme d'attention

## Transformers


[![image.png](https://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


Quelques visus et explications :

- [Seq2seq avec attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


```python
class TransformerTagger(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_dim: int,
        hidden_size: int,
        n_classes: int,
        num_layers: int=1,
        num_heads: int=1,
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size+1, embeddings_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=embeddings_dim,
            dim_feedforward=hidden_size,
            nhead=num_heads,
            batch_first=True,
        )
        self.hidden = torch.nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output = torch.nn.Linear(embeddings_dim, n_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        emb = self.embeddings(inpt)
        emb = emb.view(1, emb.shape[0], emb.shape[1])
        hid = self.hidden(emb)
        hid = hid.view(hid.shape[1], hid.shape[2])
        out = self.output(hid)
        return self.softmax(out)
    
    def predict(self, tokens: Sequence[str]) -> Sequence[str]:
        """Predict the POS for a tokenized sequence"""
        words_idx = encode(tokens)
        with torch.no_grad():
            out = self(words_idx)
        out_predictions = out.argmax(dim=-1)
        return get_pos_names(out_predictions)

transformer_tagger = TransformerTagger(len(idx_to_token), 128, 256, len(upos_names))
transformer_tagger.predict(["Le", "petit", "chat", "est", "content"])
```

```python
torch.manual_seed(0)
random.seed(0)
transformer_tagger = TransformerTagger(
    len(idx_to_token),
    128,
    256,
    len(upos_names),
    num_layers=1,
    num_heads=1,
)
train_network(
    transformer_tagger,
    [vectorize(row) for row in train_dataset],
    [vectorize(row) for row in dataset["validation"]],
    8,
)
```

## Représentations vectorielles contextuelles

```python
import transformers
```

### Tokenization en sous-mots

```python
tok = transformers.AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
tok.tokenize("Morgan reconnait l'existence du kiwi.")
```

### Obtenir des représentations vectorielles

```python
model = transformers.AutoModel.from_pretrained("flaubert/flaubert_small_cased")
with torch.no_grad():
    embeddings = model(**tok("Morgan reconnait l'existence du kiwi.", return_tensors="pt")).last_hidden_state
display(embeddings)
display(embeddings.shape)
```

### Représentations contextuelles

```python
display(tok.tokenize("Alex a de riches idées."))
display(tok.tokenize("Mangez les riches!"))
```

```python
with torch.no_grad():
    embeddings = model(**tok("Alex a eu une riche idée", return_tensors="pt")).last_hidden_state
    other_embeddings = model(**tok("Mangez les riches!", return_tensors="pt")).last_hidden_state
display(embeddings[0, 3, :8])
display(other_embeddings[0, 3, :8])
```

```python
display(tok.tokenize("Morgan reconnait Keltie."))
display(tok.tokenize("Morgan reconnait sa mère."))
with torch.no_grad():
    embeddings = model(**tok("Morgan reconnait Keltie.", return_tensors="pt")).last_hidden_state
    other_embeddings = model(**tok("Morgan reconnait sa mère.", return_tensors="pt")).last_hidden_state
display(embeddings[0, 1, :8])
display(other_embeddings[0, 1, :8])
```

### Modèles de langues masqués

```python
lm = transformers.pipeline("fill-mask", model="flaubert/flaubert_small_cased")
```

```python
lm(f"En France, c'est {lm.tokenizer.mask_token} qui est la meilleure université.")
```
