from datasets import Features, Sequence, Value
import evaluate
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer
from torchmetrics import TranslationEditRate

# patching tokenization requires access to private methods
from torchmetrics.functional.text.helper import _validate_inputs
from torchmetrics.functional.text.ter import (
    _compute_sentence_statistics,
    _compute_ter_score_from_statistics,
)

class TexEditRate(TranslationEditRate):
    """Adapt torchmetrics TranslationEditRate for TeX"""
    def __init__( self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()

    def _tokenize(self, text):
        tokens = list()
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and not tokentype is Comment:
                if any(tokentype is tp for tp in [Text, Name.Attribute, Name.Builtin]):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())

        return tokens

    def update(self, preds, target):
        """Overwrite update method from TranslationEditRate to support TeX tokenization"""
        target, preds = _validate_inputs(target, preds)

        for (pred, tgt) in zip(preds, target):
            tgt_words_ = [self._tokenize(_tgt) for _tgt in tgt]
            pred_words_ = self._tokenize(pred)
            num_edits, tgt_length = _compute_sentence_statistics(pred_words_, tgt_words_)
            self.total_num_edits += num_edits
            self.total_tgt_len += tgt_length
            if self.sentence_ter is not None:
                self.sentence_ter.append(_compute_ter_score_from_statistics(num_edits, tgt_length).unsqueeze(0))


class TER(evaluate.Metric):
    """Translation Edit Rate for LaTeX, wrapper around torchmetrics"""

    def __init__( self, **kwargs):
        super().__init__(**kwargs)
        self.ter = TexEditRate()

    def _info(self):
        return evaluate.MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(dict(
                references=Sequence(Value("string")),
                predictions=Value("string"),
            )),
        )

    def _compute(self, references, predictions):
        self.ter.update(predictions, references)
        score = self.ter.compute().item() # type: ignore
        self.ter.reset()

        return {
            "TER": score
        }
