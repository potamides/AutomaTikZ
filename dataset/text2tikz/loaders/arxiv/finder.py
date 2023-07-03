#!/usr/bin/env python

from collections import namedtuple
from functools import cached_property
from re import DOTALL, findall, search

from TexSoup import TexSoup
from .demacro import TexDemacro, Error as DemacroError

class TikzFinder():
    """
    Find tikzpictures and associated captions in a latex document and extract
    them as minimal compileable documents. Uses a combination of regex (fast)
    and TexSoup (slow) for searching.
    """
    Tikz = namedtuple("TikZ", ['code', 'caption'])
    Preamble = namedtuple("Preamble", ['imports', 'macros'])

    def __init__(self, tex):
        self.tex = self._check(tex.strip())

    def _check(self, tex):
        assert r"\documentclass" in tex, "No documentclass found!"
        assert r"\begin{document}" in tex, "No document found!"
        assert r"\end{document}" in tex, "File seems to be incomplete!"
        return tex

    @cached_property
    def _preamble(self) -> "Preamble":
        """
        Extract relevant package imports and possible macros from the document preamble.
        """
        # Patterns for the most common stuff to retain in a (tikz) document (\usepackage, \usetikzlibrary, \tikzset, etc).
        include = ["documentclass", "tikz", "tkz", "pgf", "inputenc", "fontenc", "fontspec", "amsmath", "amssymb"]
        # hard exclude macros ([re]newcommand, [re]newenvironment), as they are handled by de-macro
        exclude = [r"\new", r"\renew"]
        preamble, *_ = self.tex.partition(r"\begin{document}")

        try:
            # try TexSoup first, as it works with multiline statements
            soup = TexSoup(preamble, tolerance=1)
            statements = map(str, soup.children)
        except:
            statements = preamble.split("\n")

        tikz_preamble, maybe_macros = list(), list()
        for stmt in statements:
            if not stmt.lstrip().startswith("%"): # filter line comments
                if not any(stmt.lstrip().startswith(pat) for pat in exclude) and any(pat in stmt for pat in include):
                    tikz_preamble.append(stmt)
                else:
                    maybe_macros.append(stmt)

        return self.Preamble(imports="\n".join(tikz_preamble).strip(), macros="\n".join(maybe_macros).strip())

    def _process_macros(self, macros, tikz, expand=True):
        try:
            ts = TexDemacro(macros=macros)
            return ts.process(tikz) if expand else "\n\n".join(ts.find(tikz)).strip()
        except (DemacroError, RecursionError, TypeError):
            return tikz if expand else ""

    def _make_document(self, tikz: str) -> str:
        # if the tikzpicture uses some macros, append them to the tikz preamble
        macros = self._process_macros(self._preamble.macros, tikz, expand=False)
        extended_preamble = self._preamble.imports + (f"\n\n{macros}" if macros else "")

        return "\n\n".join([extended_preamble, r"\begin{document}", tikz, r"\end{document}"])

    def _clean_caption(self, caption: str) -> str:
        # expand any macros
        caption = self._process_macros(self._preamble.macros, caption)

        try:
            cap_soup = TexSoup(caption, tolerance=1)
            # remove any labels
            for label in cap_soup.find_all("label"):
                label.delete() # type: ignore
            # convert the caption to plaintext (e.g. \textbf{bla} -> bla)
            caption = "".join(item for item in cap_soup.text)
        except:
            pass

        return " ".join(caption.split())

    def _find_caption(self, figure: str) -> str:
        """
        Captions need special handling, since we can't express balanced
        parentheses in regex.
        """
        (*_, raw_caption), caption, unmatched_parens = figure.partition(r"\caption{"), "", 1

        for c in raw_caption:
            if c == '}':
                unmatched_parens -= 1
            elif c == '{':
                unmatched_parens += 1
            if not unmatched_parens:
                break
            caption += c

        return caption

    def find(self):
        for figure in findall(r"\\begin{figure}(.*?)\\end{figure}", self.tex, DOTALL):
            if figure.count(r"\begin{tikzpicture}") == 1: # multiple tikzpictures (e.g. subfig) are above my paygrade
                if tikz := search(r"(\\begin{tikzpicture}.*\\end{tikzpicture})", figure, DOTALL):
                    if caption := self._find_caption(figure):
                        yield self.Tikz(self._make_document(tikz.group(1)), self._clean_caption(caption))

    def __call__(self, *args, **kwargs):
        yield from self.find(*args, **kwargs)
