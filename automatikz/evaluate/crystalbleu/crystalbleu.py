from collections import Counter
from hashlib import md5
from itertools import chain, tee
from pickle import dump, load

from crystalbleu import corpus_bleu
from datasets import Features, Sequence, Value
import evaluate
from evaluate.config import HF_EVALUATE_CACHE
from evaluate.utils.logging import get_logger
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer

logger = get_logger("evaluate")

# adopted from nltk
def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

# adopted from nltk
def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

class CrystalBLEU(evaluate.Metric):
    """Wrapper around https://github.com/sola-st/crystalbleu (adapted for LaTeX)"""

    def __init__(self, corpus, k=500, n=4, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.use_cache = use_cache
        self.trivially_shared_ngrams = self.get_trivially_shared_ngrams(corpus, k, n)

    def _info(self):
        return evaluate.MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(dict(
                references=Sequence(Value("string")),
                predictions=Value("string"),
            )),
        )

    def get_trivially_shared_ngrams(self, corpus, k, n):
        """
        Computes trivially shared ngrams and caches them.
        """
        cache_dir = HF_EVALUATE_CACHE / self.__class__.__name__.lower()
        cache_dir.mkdir(parents=True, exist_ok=True)

        dhash = md5()
        dhash.update(str(sorted(corpus)).encode())
        hashname = f"{dhash.hexdigest()}.pkl"

        if (cache_file:=(cache_dir / hashname)).is_file() and self.use_cache:
            logger.info(f"Found cached trivially shared ngrams ({cache_file})")
            with open(cache_file, "rb") as f:
                return load(f)
        else:
            all_ngrams = list()
            for o in range(1, n+1):
                for tex in corpus:
                    all_ngrams.extend(ngrams(self._tokenize(tex), o))
            frequencies = Counter(all_ngrams)

            trivially_shared_ngrams = dict(frequencies.most_common(k))
            if self.use_cache:
                logger.info(f"Caching trivially shared ngrams ({cache_file})")
                with open(cache_file, "wb") as f:
                    dump(trivially_shared_ngrams, f)
            return trivially_shared_ngrams

    def _tokenize(self, text):
        tokens = list()
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and not tokentype is Comment:
                if any(tokentype is tp for tp in [Text, Name.Attribute, Name.Builtin]):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())

        return tokens

    def _compute(self, references, predictions):
        return {
            "CrystalBLEU": 100 * corpus_bleu(
                list_of_references=[[self._tokenize(ref) for ref in refs] for refs in references],
                hypotheses=[self._tokenize(c) for c in predictions],
                ignoring=self.trivially_shared_ngrams
            )
        }
