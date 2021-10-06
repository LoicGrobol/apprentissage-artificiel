import sys
import pathlib
import re

from collections import Counter

import numpy as np

p = pathlib.Path(sys.argv[1])

file_dicts = []

for f in p.glob('*'):
    with open(f) as in_stream:
        d = Counter(re.split(r'\W+', in_stream.read()))
        file_dicts.append(d)

voc = []
for d in file_dicts:
    voc.extend(d.keys())

voc = sorted(set(voc))

array_list = []
for d in file_dicts:
    d_size = sum(d.values())
    l = [d[word]/d_size for word in voc]
    array_list.append(l)

docs_array = np.array(array_list)

tf = docs_array
idf = np.log((docs_array.shape[0]/np.count_nonzero(docs_array, axis=0)))

tfidf = tf * idf

np.savetxt(sys.stdout, tfidf, delimiter='\t')
