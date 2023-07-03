from statistics import mean
from typing import Optional

from datasets import Features, Image, Value
import evaluate
from torch import clamp, float16, split
from torch.cuda import current_device, is_available as has_cuda
from transformers import BatchEncoding, CLIPModel, CLIPProcessor

from scidraw.util import set_verbosity

class CLIPScore(evaluate.Metric):
    """CLIPScore for text-image or image-image evaluation."""

    def __init__(
        self,
        clip_model: str | CLIPModel = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
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

    def _batch(self, data: BatchEncoding, batch_size: int):
        for batch in  zip(*[split(value, batch_size) for value in data.values()]):
            yield BatchEncoding(dict(zip(data.keys(), batch)))

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
        scores = num_filtered * [0] # rate filtered images as lowest score zero

        if pred_filter:
            pred_proc = img_process(pred_filter)
            ref_proc = img_process(ref_filter) if self.image_to_image else txt_process(ref_filter)

            bs = self.batch_size or len(ref_proc)
            for real, fake in zip(self._batch(ref_proc, bs), self._batch(pred_proc, bs)): # type: ignore
                fake_features = self.model.get_image_features(**fake) # type: ignore
                if self.image_to_image:
                    real_features = self.model.get_image_features(**real) # type: ignore
                else:
                    real_features = self.model.get_text_features(**real) # type: ignore

                # cosine similarity between feature vectors
                fake_features = fake_features / fake_features.norm(p=2, dim=-1, keepdim=True)
                real_features = real_features / real_features.norm(p=2, dim=-1, keepdim=True)
                scores.extend(clamp((real_features * fake_features).sum(axis=-1), 0).tolist())

        return {
            "CLIPScore" + (" (img2img)" if self.image_to_image else ""): 100 * mean(scores)
        }
