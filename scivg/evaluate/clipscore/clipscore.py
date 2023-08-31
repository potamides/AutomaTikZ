from statistics import mean
from typing import Optional

from datasets import Features, Image, Value
import evaluate
from torch import float16
from itertools import islice
from torch.cuda import current_device, is_available as has_cuda
from transformers import CLIPModel, CLIPProcessor

from scivg.util import set_verbosity

class CLIPScore(evaluate.Metric):
    """CLIPScore for text-image or image-image evaluation."""

    def __init__(
        self,
        clip_model: str | CLIPModel = "openai/clip-vit-large-patch14-336",
        image_to_image: bool = False,
        device: int = current_device() if has_cuda() else -1,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        self.image_to_image = image_to_image
        super().__init__(**kwargs)

        if isinstance(clip_model, str):
            with set_verbosity("error"):
                clip_model = CLIPModel.from_pretrained(clip_model, torch_dtype=float16) # type: ignore

        self.model = clip_model.to(device) # type: ignore
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(clip_model.config.name_or_path) # type: ignore
        self.batch_size = batch_size
        self.device = device

    def _info(self):
        return evaluate.MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(dict(
                references=Image() if self.image_to_image else Value("string"),
                predictions=Image(),
            )),
        )

    # https://stackoverflow.com/a/22045226
    def _batch(self, it, size):
       it = iter(it)
       return iter(lambda: tuple(islice(it, size)), ())

    def _filter(self, references, predictions):
        pred_filter, ref_filter = list(), list()
        for pred, ref in zip(predictions, references):
            if pred: # filter 'None' values
                pred_filter.append(pred)
                ref_filter.append(ref)

        return pred_filter, ref_filter, (len(references) - len(ref_filter))

    def _compute(self, references, predictions):
        txt_process = lambda txt: self.processor(text=txt, return_tensors="pt", padding=True, truncation=True).to(self.device) # type: ignore
        img_process = lambda img: {k: v.type(self.model.dtype) for k, v in self.processor(images=img, return_tensors="pt").to(self.device).items()} # type: ignore


        pred_filter, ref_filter, num_filtered = self._filter(references, predictions)
        scores = num_filtered * [-1] # rate filtered images as lowest score

        if pred_filter:
            bs = self.batch_size or len(ref_filter)
            for real, fake in zip(self._batch(ref_filter, bs), self._batch(pred_filter, bs)): # type: ignore
                fake_proc = img_process(fake)
                real_proc = img_process(real) if self.image_to_image else txt_process(real)

                fake_features = self.model.get_image_features(**fake_proc) # type: ignore
                if self.image_to_image:
                    real_features = self.model.get_image_features(**real_proc) # type: ignore
                else:
                    real_features = self.model.get_text_features(**real_proc) # type: ignore

                # cosine similarity between feature vectors
                fake_features = fake_features / fake_features.norm(p=2, dim=-1, keepdim=True)
                real_features = real_features / real_features.norm(p=2, dim=-1, keepdim=True)
                scores.extend((real_features * fake_features).sum(axis=-1).tolist())

        return {
            "CLIPScore" + (" (img2img)" if self.image_to_image else ""): 100 * max(mean(scores), 0)
        }
