# DaTi*k*Z Dataset
This subproject can be used to recreate the DaTi*k*Z dataset from scratch.

## Installation
DaTi*k*Z relies on a full [TeX Live](https://www.tug.org/texlive) installation,
and requires [ghostscript](https://www.ghostscript.com) and
[poppler](https://poppler.freedesktop.org). Python dependencies can be
installed as follows:
```sh
pip install -r requirements.txt
```
Processing of [arXiv](https://arxiv.org) source files requires additional setup
steps. First, you need to preprocess arXiv bulk data using
[arxiv-latex-extract](https://github.com/potamides/arxiv-latex-extract). Then,
set the `DATIKZ_ARXIV_FILES` environment variable to a colon-separated list of
paths to the jsonl-files created with arxiv-latex-extract or archives
containing them.

## Usage
The whole dataset creation pipeline can be started by executing `main.py`. If
successful, it should create the following three output files:
* `datikz-raw.parquet`: Raw dataset, without splitting and caption
  augmentation.
* `datikz-train.parquet`: Train split of DaTi*k*Z.
* `datikz-test.parquet`: Test split of DaTi*k*Z with 1k items.
Note that, as the sources we crawl get continuously updated, the created
datasets will likely slightly differ from the ones we created.
