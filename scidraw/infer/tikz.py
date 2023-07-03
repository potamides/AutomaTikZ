from collections import namedtuple
from io import BytesIO
from functools import cache
from os.path import isfile
from subprocess import DEVNULL, run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional, Union
import warnings

from PIL import Image, ImageOps
import requests
import torch
from torch.cuda import current_device, is_available as has_cuda
from transformers import (
    DisjunctiveConstraint,
    PhrasalConstraint,
    Text2TextGenerationPipeline as T2TGP,
    TextGenerationPipeline as TGP,
    TextStreamer,
)
from transformers.utils import logging
from transformers.utils.hub import is_remote_url

from ..util import optional_dependencies

with optional_dependencies():
    from pdf2image.pdf2image import convert_from_bytes
    from pdfCropMargins import crop

logger = logging.get_logger("transformers")

class PdfDocument:
    def __init__(self, raw: bytes):
        self.raw = raw

class TikzDocument:
    """
    Faciliate some operations with TikZ code. To compile the images a full
    TeXLive installation is assumed to be on the PATH. Cropping additionally
    requires Ghostscript, and rasterization needs poppler (apart from the 'pdf'
    optional dependencies).
    """
     # engines to try, could also try: https://tex.stackexchange.com/a/495999
    engines = ["lualatex", "pdflatex", "xelatex"]
    Output = namedtuple("Output", ['pdf', 'status'], defaults=[None, -1])

    def __init__(self, code: str, timeout=120):
        self.code = code
        self.timeout = timeout

    @property
    def status(self) -> int:
        return self.compile().status

    @property
    def pdf(self) -> Optional[PdfDocument]:
        return self.compile().pdf

    def compiled_with_error(self):
        return self.status != 0

    @cache
    def compile(self) -> "Output":
        output = dict()
        with TemporaryDirectory() as tmpdirname:
            with NamedTemporaryFile(dir=tmpdirname, buffering=0) as tmpfile:
                codelines = self.code.split("\n")
                codelines.insert(1, r"\thispagestyle{empty}") # make sure we don't have page numbers in compiled pdf (for cropping)
                tmpfile.write("\n".join(codelines).encode())

                # compile
                for engine in self.engines:
                    kwargs = dict(stdout=DEVNULL, stderr=DEVNULL, timeout=self.timeout, cwd=tmpdirname)
                    output['status'] = run([engine, "-interaction=nonstopmode", tmpfile.name], **kwargs).returncode
                    if output['status'] == 0:
                        break

                # crop
                if isfile(pdfname:=f"{tmpfile.name}.pdf"):
                    croppedname = f"{tmpfile.name}.crop"
                    crop(["-c", "gb", "-p", "0", "-g", "1", "-o", croppedname, pdfname], quiet=True)
                    with open(croppedname, "rb") as pdf:
                        output['pdf'] = PdfDocument(pdf.read())

        return self.Output(**output)

    def rasterize(self, size=224, expand_to_square=True) -> Optional[Image.Image]:
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
        temperature=0.2, # generation parameters based on results from codeparrot
        top_p=0.95,
        num_beams=1,
        stream=False,
        expand_to_square=False,
    ):
        self.enc_dec = model.config.is_encoder_decoder
        self.expand_to_square = expand_to_square
        self.processor = getattr(tokenizer, "image", None)
        self.pipeline = (T2TGP if self.enc_dec else TGP)(
            model=model,
            tokenizer=getattr(tokenizer, "text", tokenizer),
            device=current_device() if has_cuda() else -1,
        )
        self.pipeline.model = torch.compile(model) # type: ignore

        self.default_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=1,
            max_length=self.pipeline.tokenizer.model_max_length, # type: ignore
            do_sample=True,
            clean_up_tokenization_spaces=True,
            #remove_invalid_values=True,
            return_full_text=False,
            streamer=TextStreamer(self.pipeline.tokenizer, # type: ignore
                skip_prompt=True,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True
            ),
        )

        if self.enc_dec:
            self.default_kwargs.pop("return_full_text")
        if not stream:
            self.default_kwargs.pop("streamer")

    def generate(self, caption: str, image: Optional[Union[Image.Image, str]]=None, **gen_kwargs):
        """
        Generate TikZ for a given caption.
            caption: the caption
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
        if not self.enc_dec:
            # we add a sep token for decoder only models
            caption = caption + tokenizer.sep_token # type: ignore

        # Suppress warning about using the pipeline sequentially (we are doing that on purpose!)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TikzDocument(self.pipeline( # type: ignore
                caption,
                **(self.default_kwargs | gen_kwargs),
            )[0]["generated_text"].strip()) # type: ignore

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
