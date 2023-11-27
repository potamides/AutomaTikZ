from collections import namedtuple
from functools import cache, cached_property
from io import BytesIO
from os import environ
from os.path import isfile, join
from re import MULTILINE, escape, search, sub
from subprocess import CalledProcessError, DEVNULL, TimeoutExpired
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional, Union
import warnings

from PIL import Image, ImageOps
import requests
import torch
from torch.cuda import current_device, is_available as has_cuda
from transformers import TextGenerationPipeline as TGP, TextStreamer
from transformers.utils import logging
from transformers.utils.hub import is_remote_url

from ..util import optional_dependencies, check_output

with optional_dependencies():
    from pdf2image.pdf2image import convert_from_bytes
    from pdfCropMargins import crop
    import fitz

logger = logging.get_logger("transformers")

class PdfDocument:
    def __init__(self, raw: bytes):
        self.raw = raw

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(self.raw)


class TikzDocument:
    """
    Faciliate some operations with TikZ code. To compile the images a full
    TeXLive installation is assumed to be on the PATH. Cropping additionally
    requires Ghostscript, and rasterization needs poppler (apart from the 'pdf'
    optional dependencies).
    """
     # engines to try, could also try: https://tex.stackexchange.com/a/495999
    engines = ["pdflatex", "lualatex", "xelatex"]
    Output = namedtuple("Output", ['pdf', 'status', 'log'], defaults=[None, -1, ""])

    def __init__(self, code: str, timeout=120):
        self.code = code
        self.timeout = timeout

    @property
    def status(self) -> int:
        return self.compile().status

    @property
    def pdf(self) -> Optional[PdfDocument]:
        return self.compile().pdf

    @property
    def log(self) -> str:
        return self.compile().log

    @property
    def compiled_with_errors(self) -> bool:
        return self.status != 0

    @cached_property
    def has_content(self) -> bool:
        """true if we have an image that isn't empty"""
        return (img:=self.rasterize()) is not None and img.getcolors(1) is None

    @classmethod
    def set_engines(cls, engines: Union[str, list]):
        cls.engines = [engines] if isinstance(engines, str) else engines

    @cache
    def compile(self) -> "Output":
        output = dict()
        with TemporaryDirectory() as tmpdirname:
            with NamedTemporaryFile(dir=tmpdirname, buffering=0) as tmpfile:
                codelines = self.code.split("\n")
                # make sure we don't have page numbers in compiled pdf (for cropping)
                codelines.insert(1, r"{cmd}\AtBeginDocument{{{cmd}}}".format(cmd=r"\thispagestyle{empty}\pagestyle{empty}"))
                tmpfile.write("\n".join(codelines).encode())

                try:
                    # compile
                    errorln, tmppdf, outpdf = 0, f"{tmpfile.name}.pdf", join(tmpdirname, "tikz.pdf")
                    open(f"{tmpfile.name}.bbl", 'a').close() # some classes expect a bibfile

                    def try_save_last_page():
                        try:
                            doc = fitz.open(tmppdf) # type: ignore
                            doc.select([len(doc)-1])
                            doc.save(outpdf)
                        except:
                            pass

                    for engine in self.engines:
                        try:
                            check_output(
                                cwd=tmpdirname,
                                timeout=self.timeout,
                                stderr=DEVNULL,
                                env=environ | dict(max_print_line="1000"), # improve formatting of log
                                args=["latexmk", "-f", "-nobibtex", "-norc", "-file-line-error", "-interaction=nonstopmode", f"-{engine}", tmpfile.name]
                            )
                        except (CalledProcessError, TimeoutExpired) as proc:
                            log = getattr(proc, "output", b'').decode(errors="ignore")
                            error = search(rf'^{escape(tmpfile.name)}:(\d+):.+$', log, MULTILINE)
                            # only update status and log if first error occurs later than in previous engine
                            if (linenr:=int(error.group(1)) if error else 0) > errorln:
                                errorln = linenr
                                output.update(status=getattr(proc, 'returncode', -1), log=log)
                                try_save_last_page()
                        else:
                            output.update(status=0, log='')
                            try_save_last_page()
                            break

                    # crop
                    croppdf = f"{tmpfile.name}.crop"
                    crop(["-gsf", "-c", "gb", "-p", "0", "-a", "-1", "-o", croppdf, outpdf], quiet=True)
                    if isfile(croppdf):
                        with open(croppdf, "rb") as pdf:
                            output['pdf'] = PdfDocument(pdf.read())

                except (FileNotFoundError, NameError) as e:
                    logger.error("Missing dependencies: " + (
                        "Install this project with the [pdf] feature name!" if isinstance(e, NameError)
                        else "Did you install TeX Live?"
                    ))
                except RuntimeError: # pdf error during cropping
                    pass

        if output.get("status") == 0 and not output.get("pdf", None):
            logger.warning("Could compile document but something seems to have gone wrong during cropping!")

        return self.Output(**output)

    def rasterize(self, size=336, expand_to_square=True) -> Optional[Image.Image]:
        if self.pdf:
            image = convert_from_bytes(self.pdf.raw, size=size, single_file=True)[0]
            if expand_to_square:
                image = ImageOps.pad(image, (size, size), color='white')

            return image

    def save(self, filename: str, *args, **kwargs):
        match filename.split(".")[-1]:
            case "tex": content = self.code.encode()
            case "pdf": content = getattr(self.pdf, "raw", bytes())
            case fmt if img := self.rasterize(*args, **kwargs):
                img.save(imgByteArr:=BytesIO(), format=fmt)
                content = imgByteArr.getvalue()
            case fmt: raise ValueError(f"Couldn't save with format '{fmt}'!")

        with open(filename, "wb") as f:
            f.write(content)


class TikzGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.8, # based on "a systematic evaluation of large language models of code"
        top_p: float = 0.95,
        top_k: int = 0,
        stream: bool = False,
        expand_to_square: bool = False,
        clean_up_output: bool = True,
    ):
        self.expand_to_square = expand_to_square
        self.clean_up_output = clean_up_output
        self.processor = getattr(tokenizer, "image", None)
        self.pipeline = TGP(
            model=model,
            tokenizer=getattr(tokenizer, "text", tokenizer),
            **({} if hasattr(model, "hf_device_map") else {"device": current_device() if has_cuda() else -1}),
        )
        self.pipeline.model = torch.compile(model) # type: ignore

        self.default_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            max_length=self.pipeline.tokenizer.model_max_length, # type: ignore
            do_sample=True,
            return_full_text=False,
            streamer=TextStreamer(self.pipeline.tokenizer, # type: ignore
                skip_prompt=True,
                skip_special_tokens=True
            ),
        )

        if not stream:
            self.default_kwargs.pop("streamer")

    @property
    def is_multimodal(self):
        return self.processor is not None

    def _run_pipeline(self, prompt: str, **kwargs):
        tokenizer = self.pipeline.tokenizer
        text = self.pipeline(prompt, **kwargs)[0]["generated_text"] # type: ignore

        if self.clean_up_output:
            for token in reversed(tokenizer.tokenize(prompt)): # type: ignore
                # remove leading characters because skip_special_tokens in pipeline
                # adds unwanted prefix spaces if prompt ends with a special tokens
                if text and text[0].isspace() and token in tokenizer.all_special_tokens: # type: ignore
                    text = text[1:]
                else:
                    break

            # occasionally observed artifacts
            artifacts = {
                r'\bamsop\b': 'amsopn'
            }
            for artifact, replacement in artifacts.items():
                text = sub(artifact, replacement, text) # type: ignore

        return text

    def generate(self, caption: str, snippet: str = "", image: Optional[Union[Image.Image, str]] = None, **gen_kwargs):
        """
        Generate TikZ for a given caption.
            caption: the caption
            snippet: already existing code snippet to complete
            image: optional input fed into CLIP, defaults to the caption (can be a Pillow Image, a URI to an image, or a caption)
            gen_kwargs: additional generation kwargs (potentially overriding the default ones)
        """
        model, tokenizer = self.pipeline.model, self.pipeline.tokenizer

        if self.processor:
            if isinstance(image, str) and (isfile(image) or is_remote_url(image)):
                image = Image.open(image if isfile(image) else requests.get(image, stream=True).raw)
            if isinstance(image, Image.Image):
                if self.expand_to_square:
                    image = ImageOps.pad(image, 2 * [max(image.size)], color='white') # type: ignore
                img_inputs = self.processor(images=image, return_tensors="pt").to(model.device)
                img_inputs = {k: v.type(model.dtype) for k, v in img_inputs.items()}
            else: # make use of multi-modality of clip and encode text as image
                logger.info("Visual information provided as text, using text model from CLIP.")
                img_inputs = self.processor(text=image or caption, return_tensors="pt", truncation=True).to(model.device)

            num_patches = model.get_model().vision_tower[0].config.num_patches # type: ignore
            caption = num_patches * tokenizer.mask_token + caption # type: ignore
            gen_kwargs = gen_kwargs | dict(images=img_inputs)

        if tokenizer.add_bos_token: # type: ignore
            # unexpectedly pipeline doesn't do this automatically...
            caption = tokenizer.bos_token + caption # type: ignore
        # we add a sep token for decoder only models
        caption = caption + tokenizer.sep_token # type: ignore

        # Suppress warning about using the pipeline sequentially (we are doing that on purpose!)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TikzDocument(snippet + self._run_pipeline( # type: ignore
                prompt=caption + snippet,
                **(self.default_kwargs | gen_kwargs),
            ))

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
