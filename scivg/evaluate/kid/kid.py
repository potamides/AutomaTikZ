from typing import Optional

from PIL import Image as PILImage
from datasets import Features, Image
import evaluate
from torch import float16, nn, split
from torch.cuda import current_device, is_available as has_cuda
from torchmetrics.image.kid import KernelInceptionDistance as KID
from transformers import CLIPImageProcessor, CLIPVisionModel

from ...util import set_verbosity

class CLIPFeatureWrapper(nn.Module):
    def __init__(self, model: CLIPVisionModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values.type(self.model.dtype)).pooler_output

class KernelInceptionDistance(evaluate.Metric):
    """Wrapper around torchmetrics Kernel Inception Distance with CLIP"""

    def __init__(
        self,
        subset_size: int = 50,
        clip_model: str | CLIPVisionModel = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: int = current_device() if has_cuda() else -1,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(clip_model, str):
            with set_verbosity("error"):
                clip_model = CLIPVisionModel.from_pretrained(clip_model, torch_dtype=float16) # type: ignore

        self.model = clip_model
        self.kid = KID(feature=CLIPFeatureWrapper(clip_model.to(device)), subset_size=subset_size) # type: ignore
        self.processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(clip_model.config.name_or_path) # type: ignore
        self.device = device
        self.batch_size = batch_size

    def _info(self):
        return evaluate.MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(dict(
                references=Image(),
                predictions=Image(),
            )),
        )

    @property
    def _fallback_image(self):
        image_size = self.model.config.image_size # type: ignore
        return PILImage.new("RGB", 2 * (image_size,), "white")

    def _compute(self, references, predictions):
        # map empty images to fallback image
        predictions = [pred if pred else self._fallback_image for pred in predictions]

        references = self.processor(references, return_tensors='pt')['pixel_values'].to(self.device)
        predictions = self.processor(predictions, return_tensors='pt')['pixel_values'].to(self.device)
        bs = self.batch_size or len(references)

        for ref, pred in zip(split(references, bs), split(predictions, bs)):
            self.kid.update(ref, real=True)
            self.kid.update(pred, real=False)

        kid_mean, kid_std = self.kid.compute()
        self.kid.reset()

        return {
            "KID": kid_mean.item(),
            #"KID (std)": kid_std.item()
        }
