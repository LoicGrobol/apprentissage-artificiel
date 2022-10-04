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
