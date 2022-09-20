jupytext --to ipynb slides/lecture-*/*.md
jupyter nbconvert --execute --allow-errors --to html slides/lecture-*/*.ipynb
jupyter nbconvert --to slides slides/lecture-*/lecture-*.ipynb