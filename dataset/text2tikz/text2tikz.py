"""
Combination of various sources of tikz descriptions with aligned code.
"""

from functools import partial
from io import BytesIO
from itertools import islice
from multiprocessing.pool import ThreadPool
from os import utime
from os.path import join
from re import sub
from subprocess import CalledProcessError, DEVNULL, TimeoutExpired, run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from time import mktime
from zipfile import ZipFile, is_zipfile

from PIL import ImageOps
from datasets import Features, Image, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils import logging
from pdf2image.exceptions import PDFPageCountError
from pdf2image.pdf2image import convert_from_path
from pdfCropMargins import crop
from regex import search
from transformers import set_seed

from chatbot import WizardLM
from text2tikz.loaders import (
    janosh_tikz,
    petarv_tikz,
    tex_stackexchange_com,
    texample_net,
    tikz_net,
)

logger = logging.get_logger("transformers")
set_seed(0)

PROMPT_PREFIX, POSSIBLE_SUFFIX = '"Desired outcome:', '"'
PROMPT = "Create a clear and specific caption for an image that depicts the desired outcome of the following question. Utilize all relevant details provided in the question, particularly focusing on the visual aspects. Avoid referencing any example images or code snippets. Ensure that your caption is comprehensive and accurate:"

def batched(iterable, n):
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

# https://stackoverflow.com/a/48129136
def restore_timestamps(zipname, extract_dir):
    """zipfile doesn't preserve timestamps so we have to manually fix it"""
    if is_zipfile(zipname):
        for f in ZipFile(zipname, 'r').infolist():
            fullpath = join(extract_dir, f.filename)
            date_time = mktime(f.date_time + (0, 0, -1))
            utime(fullpath, (date_time, date_time))

lock = Lock()
def tex2img(code, size=224, timeout=300, expand_to_square=True):
    codelines = code.split("\n")
    codelines.insert(1, r"\thispagestyle{empty}") # make sure we don't have page numbers in compiled pdf (for cropping)

    def try_compile(file):
        for engine in ["lualatex", "pdflatex", "xelatex", "latex"]: # could also try: https://tex.stackexchange.com/a/495999
            try:
                return run([engine, "-interaction=nonstopmode", file], check=True, cwd=tmpdirname, stdout=DEVNULL, stderr=DEVNULL, timeout=timeout)
            except CalledProcessError:
                continue
        raise ValueError("Couldn't compile latex source.")

    with TemporaryDirectory() as tmpdirname:
        with NamedTemporaryFile(dir=tmpdirname, buffering=0) as tmpfile:
            # compile
            tmpfile.write("\n".join(codelines).encode())
            try_compile(tmpfile.name)

            # crop
            with lock: # pdfCropMargins is not threadsafe
                crop(["-p", "0", "-g", "1", "-o", pdfname := f"{tmpfile.name}-cropped.pdf", f"{tmpfile.name}.pdf"], quiet=True)
            #run(["pdfcrop", pdfname := f"{tmpfile.name}.pdf", pdfname], check=True, cwd=tmpdirname)

            # rasterize
            image = convert_from_path(pdfname, size=size, single_file=True)[0]
            img_format = image.format
            if expand_to_square:
                image = ImageOps.pad(image, (size, size), color='white')

            # save
            image.save(imgByteArr:=BytesIO(), format=img_format)

            return imgByteArr.getvalue()

class TikZConfig(builder.BuilderConfig):
    """BuilderConfig for TikZ."""

    def __init__(self, *args, bs=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = bs
        self.chatbot = WizardLM(bs=bs, prefix=PROMPT_PREFIX, model_max_length=512) # tight on memory
        self.data_urls = {
            "PetarV-/TikZ": "https://github.com/PetarV-/TikZ/archive/refs/heads/master.zip",
            "janosh/tikz": "https://github.com/janosh/tikz/archive/refs/heads/main.zip",
            "tex.stackexchange.com": "https://archive.org/download/stackexchange/tex.stackexchange.com.7z/Posts.xml",
        }
        self.generators = {
            "PetarV-/TikZ": petarv_tikz.load,
            "janosh/tikz": janosh_tikz.load,
            "texample.net": texample_net.load,
            "tikz.net": tikz_net.load,
            "pgfplots.net": tikz_net.load, # tikz.net downloader also works for this site
            "tex.stackexchange.com": lambda xml_path: tex_stackexchange_com.TeXExchangeParser(xml_path).load()
        }


class TikZ(builder.GeneratorBasedBuilder):
    """A TikZ corpus."""

    BUILDER_CONFIG_CLASS = TikZConfig

    def _info(self):
        features = {
            "caption": Value("string"),
            "code": Value("string"),
            "image": Image(),
            "uri": Value("string"),
            "origin": Value("string"),
            "date": Value("timestamp[us]"),
        }

        return DatasetInfo(
            description=str(__doc__),
            features=Features(features),
        )
    def _split_generators(self, dl_manager):
        urls_to_download = self.config.data_urls
        downloaded_files = dl_manager.download(urls_to_download)
        extracted_files = dl_manager.extract(downloaded_files)

        for name, zipname in downloaded_files.items():
            restore_timestamps(zipname, extracted_files[name])

        return [
            SplitGenerator(
                name=str(Split.TRAIN), gen_kwargs={"datasets": extracted_files}
            ),
        ]

    def _remove_comments(self, text):
        if text.lstrip().startswith("%"):
            return ""
        match = search(r"(?<![^\\]\\(\\{2})*)%", text)
        if match:
            end = match.end()
            endpos = end - 1 if not text[end - 2].strip() else end
            return text[:endpos].rstrip() + "\n"
        else:
            return text

    def _clean(self, example):
        for key, maybe_text in example.items():
            try:
                example[key] = sub(r"\r\n|\r", r"\n", maybe_text).strip() # normalize newlines
            except TypeError:
                pass
        example["code"] = "".join(self._remove_comments(line) for line in example["code"].splitlines(keepends=True))

        return example

    def _truncate(self, description):
        """
        Descriptions can be longer than 512 tokens, which leads to OOM errors. So we
        truncate them to the half of the model maximum length to leave enough
        space for generating text.
        """
        tokenizer = self.config.chatbot.pipeline.tokenizer
        return tokenizer.decode(
            tokenizer(description.strip(),
            truncation=True,
            add_special_tokens=False,
            max_length=tokenizer.model_max_length//2)['input_ids']
        ).strip()

    def _captionize(self, instruction, examples):
        instructions = [instruction] * len(examples)
        inputs = [self._truncate(ex["caption"]) for ex in examples]

        for example, caption in zip(examples, self.config.chatbot(instructions, inputs)):
            example['caption'] = (
                caption.strip().removeprefix(self.config.chatbot.prefix).removesuffix(POSSIBLE_SUFFIX)
            ).split("\n")[0].strip().removesuffix(POSSIBLE_SUFFIX).strip()
        return examples

    @classmethod
    def _compile(cls, ex):
        ex["image"] = {"path": None, "bytes": tex2img(ex["code"])}
        return ex

    def _generate_examples(self, datasets):
        all_tikz, generators = set(), self.config.generators
        skipped, idx = 0, 1

        def preprocess(load, *args, **kwargs):
            for ex in load(*args, **kwargs):
                ex = self._clean(ex)
                if ex['code'] not in all_tikz:
                    all_tikz.add(ex['code'])
                    yield ex

        def captionize(loader, prompt):
            for batch in batched(loader, self.config.bs):
                yield from self._captionize(prompt, batch)

        def skip_on_error(loader):
            nonlocal skipped
            while True:
                try:
                    yield next(loader)
                except (ValueError, PDFPageCountError, TimeoutExpired):
                    skipped += 1
                except StopIteration:
                    break

        for name, load in zip(generators.keys(), (partial(preprocess, load) for load in generators.values())):
            logger.debug("Processing examples from '%s'.", name)
            match name:
                case "PetarV-/TikZ" | "janosh/tikz": loader = load(datasets[name])
                case "tex.stackexchange.com": loader = captionize(load(datasets[name]), PROMPT)
                case "texample.net" | "tikz.net": loader = load()
                case "pgfplots.net": loader = load(base_url=f"https://{name}")
                case _: raise ValueError(f'Source "{name}" not known!')

            with ThreadPool(self.config.bs) as p:
                for example in skip_on_error(p.imap_unordered(self._compile, loader)):
                    example["origin"] = name
                    yield idx, example
                    idx += 1

        if skipped:
            logger.warn(f"Couldn't compile {skipped}/{skipped+idx-1} documents.") # pyright: ignore
