# AutomaTi*k*Z<br><sub><sup>Text-Guided Synthesis of Scientific Vector Graphics with Ti*k*Z</sup></sub>
[![arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2310.00367)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14S22x_8VohMr9pbnlkB4FqtF4n81khIh)
[![Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/nllg/AutomaTikZ)

[AutomaTi*k*Z](https://github.com/potamides/AutomaTikZ) is a software library
designed for the automatic creation of scientific vector graphics using natural
language descriptions. Generating vector graphics such as SVGs directly can be
challenging, but AutomaTi*k*Z simplifies the process by using
[Ti*k*Z](https://github.com/pgf-tikz/pgf), a well-known abstract graphics
language that can be compiled into vector graphics, as an intermediary format.
Ti*k*Z's human-oriented, high-level commands facilitate conditional language
modeling with any large language model. AutomaTi*k*Z comes with a variety of
tools for working with such models.

> [!NOTE]
> If you make use of this work please [cite](https://arxiv.org/bibtex/2310.00367) it.

## Installation
The base version of AutomaTi*k*Z, which already supports inference and
training, can be installed as regular Python package using
[pip](https://pip.pypa.io/en/stable):
```sh
pip install 'automatikz[pdf] @ git+https://github.com/potamides/AutomaTikZ'
```
For compilation of generated code, AutomaTi*k*Z additionally requires a full
[TeX Live](https://www.tug.org/texlive) installation,
[ghostscript](https://www.ghostscript.com), and, for rasterization of vector
graphics, [poppler](https://poppler.freedesktop.org).

If your goal is to run your own instance of the included [web
UI](examples/webui), clone the repository and install it in editable mode like
this, instead:
 ```sh
git clone https://github.com/potamides/AutomaTikZ
pip install -e AutomaTikZ[webui]
 ```

## Usage
As long as the required dependencies are installed, using AutomaTi*k*Z to
generate, compile, render, and save Ti*k*Z drawings is straightforward.
```python
from automatikz.infer import TikzGenerator, load

generate = TikzGenerator(*load("nllg/tikz-clima-13b"), stream=True)
caption = (
    "Visual representation of a multi-layer perceptron: "
    "an interconnected network of nodes, showcasing the structure of input, "
    "hidden, and output layers that facilitate complex pattern recognition."
)

tikzdoc = generate(caption) # streams generated tokens to stdout
tikzdoc.save("mlp.tex") # save the generated code
if tikzdoc.has_content: # true if generated tikzcode compiles to non-empty pdf
    tikzdoc.rasterize().show() # raterize pdf to a PIL.Image and show it
    tikzdoc.save("mlp.pdf") # save the generated pdf
```
More involved examples, both for inference and training, can be found in the
[examples](examples) folder.

## Model Weights
We release the following weights of fine-tuned
[LLaMA](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)
and CLiMA language models on the [Hugging Face Model
Hub](https://huggingface.co/nllg):
* CLiMA<sub>7b</sub>: [nllg/tikz-clima-7b](https://huggingface.co/nllg/tikz-clima-7b)
* CLiMA<sub>13b</sub>: [nllg/tikz-clima-13b](https://huggingface.co/nllg/tikz-clima-13b)
* LLaMA<sub>7b</sub>: [nllg/tikz-llama-7b](https://huggingface.co/nllg/tikz-llama-7b)
* LLaMA<sub>13b</sub>: [nllg/tikz-llama-13b](https://huggingface.co/nllg/tikz-llama-13b)

## Datasets
While we provide official versions of our DaTi*k*Z dataset under
[releases](https://github.com/potamides/AutomaTikZ/releases/latest) (you can
find a description of each file [here](dataset#usage)), we had to remove a
considerable portion of Ti*k*Z drawings originating from
[arXiv](https://arxiv.org), as the [arXiv non-exclusive
license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) does
not permit redistribution. We do, however, release our [dataset creation
scripts](dataset) and encourage anyone to recreate the full version of DaTi*k*Z
themselves.

## Acknowledgments
The implementation of our CLiMA model is largely based on
[LLaVA](https://github.com/haotian-liu/LLaVA).
