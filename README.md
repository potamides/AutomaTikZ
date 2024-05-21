# AutomaTi*k*Z<br><sub><sup>Text-Guided Synthesis of Scientific Vector Graphics with Ti*k*Z</sup></sub>
[![OpenReview](https://img.shields.io/badge/View%20on%20OpenReview-8C1B13?labelColor=gray&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODkuMTc5IiBoZWlnaHQ9Ijc0LjM0OSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImEiIHgyPSIxIiBncmFkaWVudFRyYW5zZm9ybT0ic2NhbGUoLTUyLjg1MiA1Mi44NTIpIHJvdGF0ZSg0Ny44MyAtLjk0IC0uNzA3KSIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIHN0b3AtY29sb3I9IiM1OGM5ZWQiIG9mZnNldD0iMCIvPjxzdG9wIHN0b3AtY29sb3I9IiM1OGM5ZWQiIG9mZnNldD0iLjE0OSIvPjxzdG9wIHN0b3AtY29sb3I9IiM2MGMzMzIiIG9mZnNldD0iLjQ3Ii8+PHN0b3Agc3RvcC1jb2xvcj0iI2NhZTMxYyIgb2Zmc2V0PSIuODI4Ii8+PHN0b3Agc3RvcC1jb2xvcj0iI2NhZTMxYyIgb2Zmc2V0PSIxIi8+PC9saW5lYXJHcmFkaWVudD48Y2xpcFBhdGggaWQ9ImIiPjxwYXRoIGQ9Ik0wIDEwOGgyMTZWMEgwWiIvPjwvY2xpcFBhdGg+PC9kZWZzPjxwYXRoIHRyYW5zZm9ybT0ibWF0cml4KDEuMzMzMyAwIDAgLTEuMzMzMyAtMS44MTMgMTAwLjE4KSIgZD0ibTE0LjcyMyA0Ni4zMDQgNC42MzEtLjAwMnYzLjk0N2w0LjkwMyA0LjkwNGg2LjM5MXY3LjA3NmgtNy4wNzVsLS4wMDEtNi4zOTEtNC45MDUtNC45MDRoLTMuOTQ0em0tMi4wNDEgMTQuODA0djEuODUxbC0xLjg1Mi0uMDAxdi0xLjQyMWwtMS41ODEtMS41ODItMi45NS0uMDAxLS4wMDEtMy4zNzdoMy4zNzlsLjAwMiAyLjk0OSAxLjU4IDEuNTgyem0yOC45MzYtMTguNjY3IDIuMjMgMi4yMzFoMi4wMDR2Mi42MDhoLTIuNjA4di0yLjAwNGwtMi4yMzEtMi4yMy00LjE1OC0uMDAydi00Ljc2bDQuNzYzLjAwMXptNi41MzgtMTAuMzM5djQuNzYxaC00Ljc2bC0uMDAyLTQuMTU4LTIuMjMtMi4yM0gzOS4xNnYtMi42MDloMi42MTF2Mi4wMDVsMi4yMjggMi4yM3pNNC43NzQgNDcuMTEybC0xLjc5OC0xLjc5NkgxLjM2di0yLjEwNWwyLjEwNC4wMDF2MS42MTZsMS43OTkgMS43OThoMy4zNXYzLjg0SDQuNzc1Wm0xNC42NzUtMTIuNjctMi41MjYtMi41MjdoLTQuNzF2LTUuMzk0aDUuMzkzbC4wMDIgNC43MDkgMi41MjggMi41MjloMi4yNjhsLjAwMSAyLjk1NGgtMi45NTZ6bTguNzI3LTIuNDM5LTIuNTI3LTIuNTI2aC0yLjI2OGwtLjAwMi0yLjk1NmgyLjk1NnYyLjI3MWwyLjUyNiAyLjUyN2g0LjcxdjUuMzk0aC01LjM5M1ptMTIuMjA4IDE5LjcxNC01LjY3NC0uMDAydi00Ljk4N2wtMi41NzEtMi41NzJoLTMuOTQ1di00LjYzMWg0LjYzdjMuOTQ2bDIuNTcxIDIuNTcxaDQuOTg5em0tOS4xOTktNS4wMTZ2NC45NzhsMy4wNTEgMy4wNTFoNS4zNzR2Ni4wNjFsLTYuMDU4LS4wMDF2LTUuMzc0bC0zLjA1My0zLjA1M2gtNC45Nzh2LTQuOTc1bC0yLjk0My0yLjk0N2gtMi44OTZsLS4wMDItMy41ODJoMy41ODJ2Mi44OThsMi45NDUgMi45NDR6TTE5LjE1OCA3Mi43MDRoLTEuMjV2LTEuMjUxaDEuMjV6bS0yLjIyMS00LjM1NWgtMy4xVjY1LjI1bDMuMS4wMDF6bS0zLjkzNSAzLjEzOWgtMi4yOTl2LTIuMjk5aDIuMjk5em0zMy4zOC00My45NDRoMi4xMTJ2Mi4xMTNoLTIuMTEyWk0xMi42ODUgNDQuNDE3aC0yLjExMXYtMi4xMTFoMi4xMTF6bTUuMzQgMTAuOTA0aC0yLjExdi0yLjExaDIuMTF6bTUuNjUxIDE0LjM4NGgtNC4zNDJ2LTQuMzQ0aDQuMzQyem0xMS4yNzItMi4wMTdoLTQuMjA1di00LjIwNGg0LjIwNXptLTE0LjIxMi00Ljc1M2gtNS41NjRWNTcuMzdsNS41NjQuMDAxem0yNy43NDEtMTEuMDMyLTIuNzk2LS4wMDJ2LTIuNzk0bDIuNzk2LjAwMXptLTQuODc1IDUuNDYzLTIuMjQ3LS4wMDEtLjAwMS0yLjI0N2gyLjI0OHpNMjguNzEgNjcuNTY5aC0yLjc0MXYtMi43NDFsMi43NDEuMDAxek0xMi40NyA1NC4yMzNoLTIuMzIydi0yLjMyMWgyLjMyMlptMjcuMDc1LTE4Ljg3MWgtMi4zMmwtLjAwMi0yLjMyM2gyLjMyNHoiIGZpbGw9InVybCgjYSkiLz48ZyB0cmFuc2Zvcm09Im1hdHJpeCgxLjMzMzMgMCAwIC0xLjMzMzMgLTEuODEzIDEwMC4xOCkiIGNsaXAtcGF0aD0idXJsKCNiKSI+PHBhdGggZD0iTTU5LjA1MSAzMS4zOTZoLTIuOTY1djIuOTY2aDIuOTY1em0tNC40OCAxMy40ODJoLTEuNjA2djEuNjA2bDEuNjA2LjAwMXptLTUuODQzIDE1LjAyN2gtMi44NDJ2Mi44NDFoMi44NDJ6bS03LjY4NCA3Ljc2NmgtMS42NTN2MS42NTFoMS42NTN6TTY2Ljg3MiAzNy41NGMtMy43NjEgMy42MDEtLjk3NiAzLjYyLS45MDIgNS44MzguMDc2IDIuMjE3LTEuNzgxIDIuNjMxLTEuNzgxIDIuNjMxLjgwMyAxLjg4Ny45MDMgMi4zMjMuOTAzIDIuMzIzLTEuNzA1IDEuMzAzLTIuMDA2IDIuOTEtMi4wMDYgMi45MSAxLjcwNCAyLjYwNyAyLjkwNCA1LjYwMiAyLjkwNCA1LjYwMnMtMTEuMjYzIDMuNjA4LTE1LjUxNCA2LjMzNGMtMS42MDQuNjY2LTEuMjAzIDMuMTQ1LTEuMjAzIDMuMTQ1LTExLjAzNCAxNC4xNDQtMjcuNTY0IDcuMDIyLTI3LjU2NCA3LjAyMnYtLjQwMnMxLjI1Ni0uMTUgMy4yNTEtLjcwOWwyLjc2OC0uMDAydi0uOTA2YTM3IDM3IDAgMCAwIDMuMjY1LTEuNDExaDYuNTJsLS4wMDItNC4xMzVhMzQgMzQgMCAwIDAgMi4xMDktMS44MzNoMy42MDFsLjAwMS0zLjk4M2EzNSAzNSAwIDAgMCAyLjU0LTMuNzc0Yy40OTQtLjc3My45NTEtMS41NDUgMS4zODEtMi4zMmgzLjIzNHYtNC41NDNsMi41ODgtLjAwMnYtMi44NGwtMS4yOTYtLjAxOHMuMDE0LTMuMDM0LjAxNS00LjQ5OGwtMy40NDQtLjAwMXYtMy40OTdoMy40OTh2My40MDZoMy41MDJ2LTYuNjEzaC0yLjYwOGMuNjQzLTkuMjQ0LTEuNjUyLTE1Ljg4OC0xLjY1Mi0xNS44ODggMS40MDUgMy45MTMgNS4xMTYgNy4xNzMgMTAuMjAxIDguOTk3IDQuNzg4IDEuNzE1IDkuNDU0IDUuNTY2IDUuNjkxIDkuMTY3IiBmaWxsPSIjNDE0ODViIi8+PC9nPjwvc3ZnPgo=)](https://openreview.net/forum?id=v3K5TVP8kZ)
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
While we provide the official version of our DaTi*k*Z dataset on the [Hugging
Face Hub](https://huggingface.co/datasets/nllg/datikz), we had to remove a
considerable portion of Ti*k*Z drawings originating from
[arXiv](https://arxiv.org), as the [arXiv non-exclusive
license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) does
not permit redistribution. We do, however, release our [dataset creation
scripts](https://github.com/potamides/DaTikZ) and encourage anyone to recreate
the full version of DaTi*k*Z themselves.

## Acknowledgments
The implementation of our CLiMA model is largely based on
[LLaVA](https://github.com/haotian-liu/LLaVA).
