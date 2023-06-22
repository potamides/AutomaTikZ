#!/usr/bin/env python

from collections import namedtuple
from functools import cached_property
from re import DOTALL, findall, search
from typing import Optional

from TexSoup import TexSoup
from TexSoup.data import TexNode
from .demacro import TexDemacro, Error as DemacroError

class TikzFinder():
    """
    Find tikzpictures and associated captions in a latex document and extract
    them as minimal compilable documents. Uses either regex (fast) TexSoup
    (potentially better results, but prohibitively slow) for searching.
    """
    Tikz = namedtuple("TikZ", ['code', 'caption'])
    Preamble = namedtuple("Preamble", ['tikz', 'macros'])

    def __init__(self, tex):
        self.tex = self._check(tex.strip())

    def _check(self, tex):
        assert r"\documentclass" in tex, "No documentclass found!"
        assert r"\begin{document}" in tex, "No document found!"
        assert r"\end{document}" in tex, "File seems to be incomplete!"
        return tex

    @cached_property
    def preamble(self) -> "Preamble":
        # Patterns for the most common stuff in a tikz document. Other stuff (e.g. macros) will be attempted to be inlined.
        patterns = ["documentclass", "tikz", "tkz", "pgf", "inputenc", "fontenc", "fontspec", "amsmath"]
        preamble, *_ = self.tex.partition(r"\begin{document}")

        try:
            # try TexSoup first, as it works with multiline statements
            soup = TexSoup(preamble, tolerance=1)
            statements = map(str, soup.children)
        except:
            statements = preamble.split("\n")

        tikz_preamble, maybe_macros = list(), list()
        for statement in statements:
            if not statement.lstrip().startswith("%"): # filter line comments
                if any(pattern in statement for pattern in patterns):
                    tikz_preamble.append(statement)
                else:
                    maybe_macros.append(statement)

        return self.Preamble(tikz="\n".join(tikz_preamble).strip(), macros="\n".join(maybe_macros).strip())

    def _expand_macros(self, macros, tikz):
        try:
            ts = TexDemacro(macros=macros)
            return ts.process(tikz)
        except (DemacroError, RecursionError, TypeError):
            return tikz

    def _make_document(self, tikz: str) -> str:
        # if there are some macros, which didn't make it into the tikz preamble, try to expand them
        macro_expanded = self._expand_macros(self.preamble.macros, tikz)
        return "\n\n".join([self.preamble.tikz, r"\begin{document}", macro_expanded, r"\end{document}"])

    def _normalize_caption(self, caption: str) -> str:
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

    def _regex_find(self):
        for figure in findall(r"\\begin{figure}(.*?)\\end{figure}", self.tex, DOTALL):
            if figure.count(r"\begin{tikzpicture}") == 1: # multiple tikzpictures (e.g. subfig) are above my paygrade
                if tikz := search(r"(\\begin{tikzpicture}.*\\end{tikzpicture})", figure, DOTALL):
                    if caption := self._find_caption(figure):
                        try:
                            cap_soup = TexSoup(caption, tolerance=1)
                            # remove any labels
                            for label in cap_soup.find_all("label"):
                                label.delete()
                            # convert the caption to plaintext (e.g. \textbf{bla} -> bla)
                            caption = "".join(item for item in cap_soup.text)
                        except:
                            pass
                        yield self.Tikz(self._make_document(tikz.group(1)), self._normalize_caption(caption))

    def _texsoup_find(self):
        soup = TexSoup(self.tex, tolerance=1)

        def find_figure(thing: TexNode) -> Optional[TexNode]:
            while thing:
                if thing.name == "figure":
                    return thing
                else:
                    thing = thing.parent
                    if thing and thing.name == "tikzpicture":
                        break

        for tikz in soup.find_all("tikzpicture"):
            caption, parent = None, tikz.parent

            if parent.name == "savebox":
                boxname = parent.args[0].contents[0].name
                for usebox in soup.find_all(boxname):
                    if "savebox" not in usebox.parent.name:
                        if figure := find_figure(usebox.parent):
                            caption = figure.caption
                            break
            elif figure := find_figure(parent):
                    caption = figure.caption

            if caption:
                caption_str = "".join(item for item in caption.text)
                if caption_str:
                    yield self.Tikz(self._make_document(str(tikz)), self._normalize_caption(caption))

    def find(self, texsoup=False):
        yield from self._texsoup_find() if texsoup else self._regex_find()

    def __call__(self, *args, **kwargs):
        yield from self.find(*args, **kwargs)
