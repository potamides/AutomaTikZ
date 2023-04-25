from transformers import (
    DisjunctiveConstraint,
    PhrasalConstraint,
    Text2TextGenerationPipeline as T2TGP,
    TextGenerationPipeline as TGP,
)
from torch.cuda import current_device, is_available as has_cuda


class TikZGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        temperature=0.2,
        max_length=1024,
        num_beams=1,
    ):
        self.enc_dec = model.config.is_encoder_decoder
        self.pipeline = (T2TGP if self.enc_dec else TGP)(
            model=model,
            tokenizer=tokenizer,
            device=current_device() if has_cuda() else -1,
        )

        self.gen_kwargs = dict(
            temperature=temperature,
            # top_p=0.9,
            num_beams=num_beams,
            #constraints=self._prepare_constraints(),
            num_return_sequences=1,
            max_length=max_length,
            do_sample=True,
            clean_up_tokenization_spaces=True,
            #remove_invalid_values=True,
        ) | ({} if self.enc_dec else dict(return_full_text=False))

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


    def generate(self, instruction):
        tokenizer = self.pipeline.tokenizer
        return self.pipeline(
            instruction + ("" if self.enc_dec else tokenizer.sep_token),
            **self.gen_kwargs,
        )[0]["generated_text"].strip()

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
