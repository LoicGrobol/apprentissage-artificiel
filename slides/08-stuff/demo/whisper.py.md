---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    custom_cell_magics: kql
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: sandbox
    language: python
    name: python3
---

```python
import datasets
import IPython
import transformers
```

```python
%time pipe = transformers.pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
```

```python
ds = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "br")
```

```python
print(ds["test"][16]["sentence"])
IPython.display.Audio(ds["test"][16]["audio"]["path"])
```

```python
%time pipe(ds["test"][16]["audio"]["path"])
```

```python
%time pipe(ds["test"][16]["audio"]["path"], generate_kwargs={"language": "br"})
```

```python
%time !adskrivan {ds["test"][16]["audio"]["path"]}
```

```python
print(ds["train"][8]["sentence"])
IPython.display.Audio(ds["train"][8]["audio"]["path"])
```

```python
%time pipe(ds["train"][8]["audio"]["path"])
```

```python
medium_sents = ds["test"].filter(lambda x: x["sentence"].count(" ") > 8)
medium_sents
```

```python
print(medium_sents[8]["sentence"])
IPython.display.Audio(medium_sents[8]["audio"]["path"])
```

```python
%time pipe(medium_sents[8]["audio"]["path"], generate_kwargs={"language": "br"})
```

```python
%time !adskrivan {medium_sents[8]["audio"]["path"]}
```

```python
long_sents = ds["test"].filter(lambda x: x["sentence"].count(" ") > 16)
long_sents
```

```python
print([r["sentence"] for r in long_sents])
IPython.display.Audio(long_sents[8]["audio"]["path"])
```

```python
%time pipe(long_sents[0]["audio"]["path"], generate_kwargs={"language": "br"})
```

```python
%time !adskrivan {long_sents[8]["audio"]["path"]}
```
