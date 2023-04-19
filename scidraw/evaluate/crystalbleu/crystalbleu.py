from collections import Counter

from datasets import Features, Sequence, Value
from nltk.util import ngrams
from pygments.token import Name, Comment, Text

from crystalbleu import corpus_bleu
import evaluate
from pygments.lexers.markup import TexLexer
from sacremoses import MosesTokenizer

class CrystalBLEU(evaluate.Metric):
    """Wrapper around https://github.com/sola-st/crystalbleu (adapted for LaTeX)"""

    def __init__(self, corpus, k=500, n=4, **kwargs):
        super().__init__(**kwargs)
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.k = k

        all_ngrams = list()
        for o in range(1, n+1):
            for tex in corpus:
                all_ngrams.extend(ngrams(self._tokenize(tex), o))
        frequencies = Counter(all_ngrams)
        self.trivially_shared_ngrams = dict(frequencies.most_common(self.k))

    def _info(self):
        return evaluate.MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "references": Sequence(Value("string")),
                    "candidates": Value("string"),
                }
            ),
        )

    def _tokenize(self, text):
        tokens = list()
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and not tokentype is Comment:
                if any(tokentype is tp for tp in [Text, Name.Attribute, Name.Builtin]):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())

        return tokens

    def _compute(self, references, candidates):
        return {
            "CrystalBLEU": corpus_bleu(
                list_of_references=[[self._tokenize(ref) for ref in refs] for refs in references],
                hypotheses=[self._tokenize(c) for c in candidates],
                ignoring=self.trivially_shared_ngrams
            )
        }
