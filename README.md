[comment]: <> "LTeX: language=fr"
<!-- markdownlint-disable MD003 MD025 MD033 -->

Apprentissage artificiel
========================

[![Licence : CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

Contenus pour le cours « Apprentissage automatique » du master [Plurital](http://plurital.org).


- [Site du cours](https://loicgrobol.github.io/apprentissage-artificiel/)
- [Dépôt GitHub](https://github.com/LoicGrobol/apprentissage-artificiel)

Contact : [<loic.grobol@parisnanterre.fr>](mailto:loic.grobol@parisnanterre.fr)

## Générer le site en local

Très ad-hoc pour l'instant, ca s'arrangera

```console
uv pip install --upgrade --requirements requirements.lst
uv pip install --upgrade --requirements src/notebooks/requirements.lst
```

Regenerate:

```bash
python tools/manage.py build "src"
```

## Licences

[![CC BY Licence badge](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)


Copyright © 2026 L. Grobol [\<morgan.grobol@tuta.com\>](mailto:morgan.grobol@tuta.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/).

Un résumé simplifié de cette licence est disponible à <https://creativecommons.org/licenses/by/4.0/>.

Le texte intégral de cette licence est disponible à
<https://creativecommons.org/licenses/by/4.0/legalcode>
