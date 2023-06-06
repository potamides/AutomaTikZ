from os.path import isfile
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

logger = logging.get_logger("transformers")

class TikZGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        temperature=0.2, # generation parameters based on results from codeparrot
        top_p=0.95,
        max_length=1024,
        num_beams=1,
        stream=True,
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
        self.pipeline.model = torch.compile(model) # pyright: ignore

        self.default_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            #constraints=self._prepare_constraints(),
            num_return_sequences=1,
            max_length=max_length,
            do_sample=True,
            clean_up_tokenization_spaces=True,
            #remove_invalid_values=True,
            return_full_text=False,
            streamer=TextStreamer(self.pipeline.tokenizer, # pyright: ignore
                skip_prompt=True,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True
            ),
        )

        if self.enc_dec:
            self.default_kwargs.pop("return_full_text")
        if not stream:
            self.default_kwargs.pop("streamer")

    def _prepare_constraints(self):
        with_prefix_space = [
            r"\documentclass"
        ]

        with_and_without_prefix_space = [
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\end{tikzpicture}",
            r"\end{document}"
        ]

        tk = lambda s: self.pipeline.tokenizer(s, add_special_tokens=False).input_ids # pyright: ignore
        cv = lambda t: self.pipeline.tokenizer.convert_tokens_to_ids(t) # pyright: ignore

        return (
            [PhrasalConstraint(tk(term)) for term in with_prefix_space] +
            [DisjunctiveConstraint([(tokens := tk(term)), [cv(term[0])] + tokens[1:]]) for term in with_and_without_prefix_space]
        )


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
                    image = ImageOps.pad(image, 2 * [max(image.size)], color='white') # pyright: ignore
                img_inputs = self.processor(images=image, return_tensors="pt").to(model.device)
                img_inputs = {k: v.type(model.dtype) for k, v in img_inputs.items()}
            else: # make use of multi-modality of clip and encode text as image
                logger.info("Visual information provided as text, using text model from CLIP.")
                img_inputs = self.processor(text=image or caption, return_tensors="pt", truncation=True).to(model.device)

            num_patches = model.get_model().vision_tower[0].config.num_patches # pyright: ignore
            caption = num_patches * tokenizer.mask_token + caption # pyright: ignore
            gen_kwargs = gen_kwargs | dict(images=img_inputs)

        if tokenizer.add_bos_token: # pyright: ignore
            # unexpectedly pipeline doesn't do this automatically...
            caption = tokenizer.bos_token + caption # pyright: ignore
        if not self.enc_dec:
            # we add a sep token for decoder only models
            caption = caption + tokenizer.sep_token # pyright: ignore

        # Suppress warning about using the pipeline sequentially (we are doing that on purpose!)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.pipeline( # pyright: ignore
                caption,
                **(self.default_kwargs | gen_kwargs),
            )[0]["generated_text"].strip() # pyright: ignore

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
