#!/usr/bin/env python

from argparse import ArgumentParser
from functools import lru_cache
from importlib.resources import files
from inspect import signature
from multiprocessing.pool import ThreadPool
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Optional

from PIL import Image
import fitz
import gradio as gr
from transformers import TextIteratorStreamer

from automatikz.infer import TikzDocument, TikzGenerator, load

assets = files(__package__) / "assets" if __package__ else files("assets") / "."
models = {
    "CLiMA-7b":  "nllg/tikz-clima-7b",
    "CLiMA-13b": "nllg/tikz-clima-13b",
    "LLaMA-7b":  "nllg/tikz-llama-7b",
    "LLaMA-13b": "nllg/tikz-llama-13b"
}

@lru_cache(maxsize=1)
def cached_load(*args, **kwargs):
    gr.Info("Instantiating model. Could take a while...") # type: ignore
    return load(*args, **kwargs)

def convert_to_svg(pdf):
    doc = fitz.open("pdf", pdf.raw) # type: ignore
    return doc[0].get_svg_image()

def inference(
    model_name: str,
    caption: str,
    image: Optional[Image.Image],
    temperature: float,
    top_p: float,
    top_k: int,
    expand_to_square: bool,
):
    generate = TikzGenerator(
        *cached_load(model_name, device_map="auto"),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        expand_to_square=expand_to_square,
    )
    streamer = TextIteratorStreamer(
        generate.pipeline.tokenizer, # type: ignore
        skip_prompt=True,
        skip_special_tokens=True
    )

    thread = ThreadPool(processes=1)
    async_result = thread.apply_async(generate, kwds=dict(caption=caption, image=image, streamer=streamer))

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text, None, False
    yield async_result.get().code, None, True

def tex_compile(
    code: str,
    timeout: int,
    rasterize: bool
):
    tikzdoc = TikzDocument(code, timeout=timeout)
    if not tikzdoc.has_content:
        if tikzdoc.compiled_with_errors:
            raise gr.Error("TikZ code did not compile!") # type: ignore
        else:
            gr.Warning("TikZ code compiled to an empty image!") # type: ignore
    elif tikzdoc.compiled_with_errors:
        gr.Warning("TikZ code compiled with errors!") # type: ignore

    if rasterize:
        yield tikzdoc.rasterize()
    else:
        with NamedTemporaryFile(suffix=".svg", buffering=0) as tmpfile:
            if pdf:=tikzdoc.pdf:
                tmpfile.write(convert_to_svg(pdf).encode())
            yield tmpfile.name if pdf else None

def check_inputs(caption: str, _: Optional[Image.Image]):
    if not caption:
        raise gr.Error("Prompt is required")

def get_banner():
    return dedent('''\
    # AutomaTi*k*Z: Text-Guided Synthesis of Scientific Vector Graphics with Ti*k*Z

    <p>
      <a style="display:inline-block" href="https://github.com/potamides/AutomaTikZ">
        <img src="https://img.shields.io/badge/View%20on%20GitHub-green?logo=github&labelColor=gray" alt="View on GitHub">
      </a>
      <a style="display:inline-block" href="https://arxiv.org/abs/2310.00367">
        <img src="https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray" alt="View on arXiv">
      </a>
      <a style="display:inline-block" href="https://colab.research.google.com/drive/14S22x_8VohMr9pbnlkB4FqtF4n81khIh">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
      </a>
      <a style="display:inline-block" href="https://huggingface.co/spaces/nllg/AutomaTikZ">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open in HF Spaces">
      </a>
    </p>
    ''')

def remove_darkness(stylable):
    """
    Patch gradio to only contain light mode colors.
    """
    if isinstance(stylable, gr.themes.Base): # remove dark variants from the entire theme
        params = signature(stylable.set).parameters
        colors = {color: getattr(stylable, color.removesuffix("_dark")) for color in dir(stylable) if color in params}
        return stylable.set(**colors)
    elif isinstance(stylable, gr.Blocks): # also handle components which do not use the theme (e.g. modals)
        stylable.load(_js="() => document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'))")
        return stylable
    else:
        raise ValueError

def build_ui(model=list(models)[0], lock=False, rasterize=False, force_light=False, lock_reason="locked", timeout=120):
    theme = remove_darkness(gr.themes.Soft()) if force_light else gr.themes.Soft()
    with gr.Blocks(theme=theme, title="AutomaTikZ") as demo: # type: ignore
        if force_light: remove_darkness(demo)
        gr.Markdown(get_banner())
        with gr.Row(variant="panel"):
            with gr.Column():
                info = (
                    "Describe what you want to generate. "
                    "Scientific graphics benefit from captions with at least 30 tokens (see examples below), "
                    "while simple objects work best with shorter, 2-3 word captions."
                )
                caption = gr.Textbox(label="Caption", info=info, placeholder="Type a caption...")
                image = gr.Image(label="Image Input (optional)", type="pil")
                label = "Model" + (f" ({lock_reason})" if lock else "")
                model = gr.Dropdown(label=label, choices=list(models.items()), value=models[model], interactive=not lock) # type: ignore
                with gr.Accordion(label="Advanced Options", open=False):
                    temperature = gr.Slider(minimum=0, maximum=2, step=0.05, value=0.8, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.95, label="Top-P")
                    top_k = gr.Slider(minimum=0, maximum=100, step=10, value=0, label="Top-K")
                    expand_to_square = gr.Checkbox(value=True, label="Expand image to square")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.ClearButton([caption])
            with gr.Column():
                with gr.Tabs() as tabs:
                    with gr.TabItem(label:="TikZ Code", id=0):
                        info = "Source code of the generated image."
                        tikz_code = gr.Code(label=label, show_label=False, info=info, interactive=False)
                    with gr.TabItem(label:="Compiled Image", id=1):
                        result_image = gr.Image(label=label, show_label=False, show_share_button=rasterize)
                    clear_btn.add([tikz_code, result_image])
        gr.Examples(examples=str(assets), inputs=[caption, tikz_code, result_image])

        events = list()
        finished = gr.Textbox(visible=False) # hack to cancel compile on canceled inference
        for listener in [caption.submit, run_btn.click]:
            generate_event = listener(
                check_inputs,
                inputs=[caption, image],
                queue=False
            ).success(
                lambda: gr.Tabs(selected=0),
                outputs=tabs, # type: ignore
                queue=False
            ).then(
                inference,
                inputs=[model, caption, image, temperature, top_p, top_k, expand_to_square],
                outputs=[tikz_code, result_image, finished]
            )

            def tex_compile_if_finished(finished, *args):
                yield from (tex_compile(*args, timeout=timeout, rasterize=rasterize) if finished == "True" else [])

            compile_event = generate_event.then(
                lambda finished: gr.Tabs(selected=1) if finished == "True" else gr.Tabs(),
                inputs=finished,
                outputs=tabs, # type: ignore
                queue=False
            ).then(
                tex_compile_if_finished,
                inputs=[finished, tikz_code],
                outputs=result_image
            )
            events.extend([generate_event, compile_event])

        model.select(lambda model_name: gr.Image(visible="clima" in model_name), inputs=model, outputs=image, queue=False)
        for btn in [clear_btn, stop_btn]:
            btn.click(fn=None, cancels=events, queue=False)
        return demo

def parse_args():
    argument_parser = ArgumentParser(
        description="Web UI for AutomaTikZ."
    )
    argument_parser.add_argument(
        "--model",
        default=list(models)[0],
        choices=list(models),
        help="Initially selected model.",
    )
    argument_parser.add_argument(
        "--lock",
        action="store_true",
        help="Whether to allow users to change the model or not.",
    )
    argument_parser.add_argument(
        "--lock_reason",
        default="locked",
        help="Additional information why model selection is locked.",
    )
    argument_parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link for the interface.",
    )
    argument_parser.add_argument(
        "--rasterize",
        action="store_true",
        help= "Whether to rasterize the generated image before displaying it."
    )
    argument_parser.add_argument(
        "--force_light",
        action="store_true",
        help= "Whether to enforce light theme (useful for vector graphics with dark text)."
    )
    argument_parser.add_argument(
        "--timeout",
        default=120,
        type=int,
        help="Allowed timeframe for compilation.",
    )
    return vars(argument_parser.parse_args())

if __name__ == "__main__":
    args = parse_args()
    share = args.pop("share")
    build_ui(**args).queue().launch(share=share)
