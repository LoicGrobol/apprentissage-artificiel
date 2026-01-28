\begin{equation}
    p_s = \frac{\text{nombre d'occurrences de la phrase dans le corpus}}{\text{taille du corpus}}
\end{equation}

Et on aurait alors un modèle de langue parfait.


Dans ce cas, ce serait facile d'évaluer notre modèle de langue imparfait à nous : imaginons que pour
cette même phrase, notre modèle donne une vraisemblance $\hat{p}_s$, on pourrait par exemple lui
donner un score $m_s$ pour cette phrase en calculant

\begin{equation}
    M_s = d(p_s, \hat{p}_s) = \left|p_s-\hat{p}_s\right|
\end{equation}


Aparté important : les barres verticales notent ici la [valeur absolue](https://fr.wikipedia.org/wiki/Valeur_absolue) :

\begin{equation}
    \left|x\right| =
        \begin{cases}
            x &\text{si x ⩾ 0}\\
            -x &\text{sinon}
        \end{cases}
\end{equation}

par exemple $\left|2713\right|=2713$ et $\left|-2713\right|=2713$.

En Python :

```python
abs(-2713)
```

C'est un outil pour rendre un truc positif, mais la propriété qui nous intéresse ici, c'est que la
valeur absolue d'une différence, c'est une
[*distance*](https://fr.wikipedia.org/wiki/Distance_(math%C3%A9matiques)) (au sens mathématique du
terme). Un truc qui dit si deux choses sont loin l'une de l'autre. Ici $M_s$, on peut l'interpréter
comme la *distance* entre $p_s$ et $\hat{p}_s$.


(C'est pour ça que j'ai écrit $d(p_s, \hat{p}_s)$ !)


Bon, donc, toujours dans ce monde idéal, on aurait alors un outil qui nous donnerait la qualité de
notre modèle pour une phrase. C'est pas très difficile de voir comment en déduire un score global :
on a qu'à faire la moyenne sur toutes les phrases possibles ! Si on imagine qu'il y a $n$ phrases
possibles dans la langue qu'on étudie (un très très grand $n$ donc) et qu'on les appelle $s_1$,
$s_2$, …, $s_n$, le score global $M$ de notre modèle, on peut par exemple décider que ce serait la
moyenne :


\begin{equation}
    m = \frac{M_{s_1} + M_{s_2} + … + M_{s_n}}{n}
\end{equation}

<!-- #region slideshow={"slide_type": "slide"} -->
## 👜 Exo : les sacs de mots 👜
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
![](bow.png)

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### 1. Faire des sacs

- Écrire un script (un fichier `.py`, quoi) qui prend en unique argument de ligne de commande un
  dossier contenant des documents (sous forme de fichier textes) et sort un fichier TSV donnant pour
  chaque document sa représentation en sac de mots (en nombre d'occurrences des mots du vocabulaire
  commun)
  - Un fichier par ligne, un mot par colonne
  - Pour itérer sur les fichiers dans un dossier on peut utiliser `for f in
    pathlib.Path(chemin_du_dossier).glob('*'):` avec le module
    [`pathlib`](https://docs.python.org/3/library/pathlib.html#module-pathlib) (il y a d'autres
    solutions…).
  - Pour récupérer des arguments en ligne de commande : [`argparse` ou
    `sys.argv`](https://docs.python.org/3/tutorial/stdlib.html#command-line-arguments)
  - Pour écrire un `array` dans un fichier TSV :
    [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html)
- Tester sur la partie positive du [mini-corpus imdb](../../data/imdb_smol.tar.gz)

Pensez à ce qu'on a vu les cours précédents pour ne pas réinventer la roue. Par exemple vous savez
tokenizer.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### 2. Faire des sacs relatifs

Modifier le script précédent pour qu'il génère des sacs de mots utilisant les fréquences relatives
plutôt que les fréquences absolues
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### 3. Faire des tfidsacs
<!-- #endregion -->

Modifier le script de précédent pour qu'il renvoie non plus les fréquences relatives de chaque mot,
mais leur tf⋅idf avec la définition suivante pour un mot $w$, un document $D$ et un corpus $C$

- $\mathrm{tf}(w, D)$ est la fréquence relative de $w$ dans $D$
- $$\mathrm{idf}(w, C) = \log\!\left(\frac{\text{nombre de documents dans $C$}}{\text{nombre de
  documents de $C$ qui contiennent $w$}}\right)$$
- $\log$ est le logarithme naturel
  [`np.log`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)
- $\mathrm{tfidf}(w, D, C) = \mathrm{tf}(w, D)×\mathrm{idf}(w, C)$

Pistes de recherche :

- L'option `keepdims` de `np.sum`
- `np.transpose`
- `np.count_nonzero`
- Regarder ce que donne `np.array([[1, 0], [2, 0]]) > 0`
