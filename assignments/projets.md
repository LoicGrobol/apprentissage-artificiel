[comment]: <> "LTeX: language=fr"

Projets Apprentissage artificiel
================================

Votre travail sera de réaliser une application, une ressource, une interface ou une bibliothèque
pour Python. Son thème devra être lié au TAL ou au traitement de données, utiliser des notions
d'apprentissage artificiel et pourra évidemment être en lien avec d'autres cours, d'autres projets
ou votre travail en entreprise (mais assurez-vous avant que ce soit OK de partager votre code avec
moi dans ce cas).

*Bien entendu, rien ne vous empêche de combiner ces options.*

Je m'attends plus à ce que vous réalisiez un projet autonome (c'est en général plus simple), mais
votre travail peut aussi prendre la forme d'un plugin/add-on/module… pour un projet existant (pensez
à spaCy par exemple) voire une contribution substantielle à projet existant (si vous faites passer
un gros *pull request* à Pytorch par exemple) mais si c'est ce que vous visez, dites le moi très en
avance et on en discute.

## Consignes

- Projet à rendre le 4 février 2021 *au plus tard*
- Projet de préférence collectif, par groupe de 2 ou 3
  - Si c'est un problème pour vous, venez me voir, tout est négociable
  - S'il y a un problème — quel qu'il soit — dans votre groupe, n'hésitez pas à m'en parler

Le rendu devra comporter :

1. Une documentation du projet traitant les points suivants :

   - Les objectifs du projet
   - Les données (origine, format, statut juridique) et les traitements opérés
     sur celles-ci
   - La méthodologie (comment vous vous êtes répartis le travail, comment vous
     avez identifié les problèmes et les avez résolus, différentes étapes du
     projet…)
   - L'implémentation ou les implémentations (modélisation le cas échéant,
     modules et/ou API utilisés, différents langages le cas échéant)
   - Les résultats (fichiers output, visualisations…) et une discussion sur ces
     résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par
     exemple)

   On attend de la documentation technique, pas une dissertation. Elle pourra
   prendre le format d'un ou plusieurs fichiers, d'un site web, d'un notebook de
   démonstration, à votre convenance

2. Le code Python et les codes annexes (JS par ex.) que vous avez produit.
   Le code *doit* être commenté. Des tests ce serait bien. **Évitez les
   notebooks**, préférez les interfaces en ligne de commande ou web (ou
   graphiques si vous êtes très motivé⋅e⋅s)

3. Les éventuelles données en input et en output (ou un échantillon si le volume
   est important)

N'hésitez pas à vous servir de git pour versionner vos projets !

## Conseils

Écrivez ! Tenez un carnet : vos questions, un compte-rendu de vos discussions,
les problèmes rencontrés, tout est bon à prendre et cela vous aidera à rédiger
la documentation finale.

## Ressources

### Données géo-localisées

Il existe beaucoup de choses pour travailler avec des données géo-localisées. Allez voir en vrac :
[Geo-JSON](http://geojson.org/), [uMap](http://umap.openstreetmap.fr/fr/) pour créer facilement des
cartes en utilisant les fonds de carte d'OpenStreetMap, [leaflet](http://leafletjs.com/) une lib JS
pour les cartes interactives, [overpass turbo](http://overpass-turbo.eu/) pour interroger facilement
les données d'OpenStreetMap (il y a une [api !](http://www.overpass-api.de/)).

### Ressources linguistiques

N'hésitez pas à aller fouiller dans [Ortolang](https://www.ortolang.fr/) ou
[Clarin](https://lindat.mff.cuni.cz/repository/xmlui/) des ressources linguistiques exploitables
librement et facilement. Vous pouvez aussi aller voir du côté de l'API twitter pour récupérer des
données (qui ne sont pas nécessairement uniquement linguistiques)

### Open Data

Quelques sources : [Paris Open Data](https://opendata.paris.fr),
[data.gouv.fr](https://data.gouv.fr), [Google dataset
search](https://toolbox.google.com/datasetsearch)