## Réduction de dimension et clustering

Point important : si vous avez un jeu de données à plein de dimension et que vous voulez

- Trouver des clusters.
- Visualiser les données.

Vous avez deux opérations à faire : réduction de dimension (pour mettre en 2d typiquement) et
clusterisation. Du coup :

> Est-ce qu'on fait la réduction de dimension avant ou après ?

**Après**

**Sauf**

Si on a beaucoup de dimensions, comme susmentionné, ça peut être intéressant de commencer par une
réduction comme une ACP pour poasser à quelques dizaines de dimensions avant de calculer les
clusters, puis de de passer en dimension 2.

## limites de K-means

## Bon mais alors j'utilise quoi ?

- Réduction de dimension simple et brutale qui n'invente pas des clusters ? **ACP**
- Trouver des clusters ? **BGMM** → penser à bien choisir les hyperparamètres, dans le doute,
  utiliser une covariance `full`
- Visualiser des données → **UMAP**, surtout si c'est des données pour lesquelles vous avez des
  classes *gold*
- CLusteriser après réduction en 2D → même pas en rêve.
