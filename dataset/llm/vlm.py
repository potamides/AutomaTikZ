from abc import ABC, abstractmethod
from functools import cache, cached_property
from os.path import isfile
from types import SimpleNamespace
from typing import Union

from PIL import Image
from datasets.utils.logging import tqdm
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import Conversation, SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
import requests
import torch
import torch.multiprocessing as mp
from transformers import (
    CLIPModel,
    CLIPProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)
from transformers.utils.hub import is_remote_url

def get_image(image):
    if isinstance(image, str) and (isfile(image) or is_remote_url(image)):
        return Image.open(image if isfile(image) else requests.get(image, stream=True).raw)
    return image

class VisionLanguageModel(ABC):
    max_new_tokens = 256
    max_prompt_length = 128

    def __init__(self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        beams=1,
     ):
        self.device = device
        self.beams = beams

    @abstractmethod
    def model(self):
        """Return the model."""

    @abstractmethod
    def processor(self):
        """Return the processor."""

    @abstractmethod
    def generate(self):
        """Generate the output."""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


class InstructBlip(VisionLanguageModel):
    model_name = "Salesforce/instructblip-vicuna-13b"

    @cached_property
    def model(self):
        model = InstructBlipForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16
        ).to(self.device) # type: ignore
        model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        return torch.compile(model)

    @cached_property
    def processor(self):
        return InstructBlipProcessor.from_pretrained(self.model_name)

    def generate(self, text: str, image: Union[Image.Image, str]):
        inputs = self.processor(
            images=get_image(image),
            text=text,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True
        ).to(self.device, torch.float16)

        outputs = self.processor.batch_decode(self.model.generate(
            **inputs,
            do_sample=True,
            num_beams=self.beams,
            num_return_sequences=self.beams,
            max_new_tokens=self.max_new_tokens,
            top_p=0.95,
            temperature=1,
            min_length=8,
            # hack to suppress yes/no answers
            begin_suppress_tokens=self.processor.tokenizer.convert_tokens_to_ids(["yes", "no", "▁yes", "▁no"])),
            skip_special_tokens=True
        )

        return [output.strip() for output in outputs] if len(outputs) > 1 else outputs[0].strip()


class Llavar(VisionLanguageModel):
    model_name = "truehealth/LLaVar"

    @cache
    def _load_pretrained_model(self):
        model_name = get_model_name_from_path(self.model_name)
        tokenizer, model, image_processor, _ = load_pretrained_model(self.model_name, None, model_name, device_map=self.device)
        return torch.compile(model.to(self.device)), image_processor, tokenizer

    @property
    def model(self):
        return self._load_pretrained_model()[0]

    @property
    def processor(self):
        _, processor, tokenizer = self._load_pretrained_model()
        return SimpleNamespace(image=processor, text=tokenizer)

    def generate(self, text: str, image: Union[Image.Image, str]):
        if self.model.config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        conv = Conversation(
            system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
                   "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
                   "Follow the instructions carefully and explain your answers in detail.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )

        #conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = self.processor.image.preprocess(get_image(image), return_tensors='pt')['pixel_values'].half().to(self.device)
        input_ids = tokenizer_image_token(prompt, self.processor.text, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.processor.text, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                use_cache=True,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=[stopping_criteria],
                num_beams=self.beams,
                num_return_sequences=self.beams,
            )

        input_token_len = input_ids.shape[1]
        outputs = list()
        for output in self.processor.text.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True):
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            outputs.append(output.strip())

        return [output for output in outputs] if len(outputs) > 1 else outputs[0]


class CapFilt():
    def __init__(self,
        clip_model="openai/clip-vit-large-patch14-336",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device=device
        self.clip_model=clip_model

    @cached_property
    def model(self):
        return CLIPModel.from_pretrained(self.clip_model, torch_dtype=torch.float16).to(self.device)

    @cached_property
    def processor(self):
        return CLIPProcessor.from_pretrained(self.clip_model, torch_dtype=torch.float16)

    def get_best(self, image, candidates):
        inputs = self.processor(
            text=candidates,
            images=get_image(image),
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        return candidates[probs.argmax().item()]


class ParallelCaptionizer():
    def __init__(
        self,
        devices=list(range(torch.cuda.device_count())),
        vlm_class=Llavar,
        clip_model="openai/clip-vit-large-patch14-336",
        samples=5,
        prompt="Write a short description for the image.",
    ):
        self.devices = devices
        self.vlm_class = vlm_class
        self.clip_model=clip_model
        self.samples=samples
        self.prompt = prompt
        self.ctx = mp.get_context("spawn")

    def _captionize_worker(self, device, img_queue, cap_queue):
        """Each subprocess will run this function on a different GPU."""
        model, capfilt = self.vlm_class(device), CapFilt(self.clip_model, device)
        captions = dict()

        while item:=img_queue.get():
            idx, image = item
            candidates = list()
            for _ in range(self.samples):
                candidates.append(model(text=self.prompt, image=image))
            captions[idx] = capfilt.get_best(image=image, candidates=candidates) if self.samples > 1 else candidates[0]
        cap_queue.put(captions)

    def captionize(self, images):
        processes, captions = list(), dict()
        img_queue = self.ctx.Queue(maxsize=len(self.devices))
        cap_queue = self.ctx.Queue(maxsize=len(self.devices))
        for device in self.devices:
            p = self.ctx.Process(
                target=self._captionize_worker,
                args=(torch.device(device), img_queue, cap_queue)
            )
            p.start()
            processes.append(p)
        for idx, image in enumerate(tqdm(images, desc="Captioning")):
            img_queue.put((idx, image))
        for _ in processes:
            img_queue.put(None)  # sentinel value to signal subprocesses to exit
            captions.update(cap_queue.get())
        for p in processes:
            p.join()  # wait for all subprocesses to finish

        return [captions[idx] for idx in range(len(images))]

    def __call__(self, *args, **kwargs):
        return self.captionize(*args, **kwargs)
