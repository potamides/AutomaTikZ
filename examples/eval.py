#!/usr/bin/env python
from argparse import ArgumentParser
from collections import ChainMap
from json import dump
from statistics import mean

from datasets import load_dataset
from evaluate.visualization import radar_plot as _radar_plot
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import set_seed

from scidraw.evaluate import load as load_metric
from scidraw.infer import TikzGenerator, load as load_model

def parse_args():
    argument_parser = ArgumentParser(
        description="Evaluate a range of fine-tuned models."
    )
    argument_parser.add_argument(
        "--batch_size",
        default=1,
        help="batch sizes for metrics that make use of CLIP",
    )
    argument_parser.add_argument(
        "--trainset",
        required=True,
        help="path to the text2tikz train set (in parquet format)",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the text2tikz test set (in parquet format)",
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save the radar plot of the results or the raw scores (if it ends with json)",
    )
    argument_parser.add_argument(
        "--path",
        nargs='+',
        metavar="MODEL=PATH",
        required=True,
        help="(multiple) key-value pairs of model names and paths/urls to models/adapters (local or hub)",
    )
    return argument_parser.parse_args()

def load_metrics(trainset, bs=1):
    crystalbleu = load_metric("crystalbleu", corpus=trainset)
    clipscore = load_metric("clipscore", batch_size=bs)
    imgscore = load_metric("clipscore", batch_size=bs, clip_model=clipscore.model, image_to_image=True) # type: ignore
    kid = load_metric("kid", batch_size=bs)
    eed = load_metric("eed")

    def compile_success_rate(predictions):
        scores = list()
        for pred in predictions:
            if pred.pdf:
                scores.append(0.5 if pred.compiled_with_errors() else 1)
            else:
                scores.append(0)
        return {"CSR": 100 * mean(scores)}

    def compute(references, predictions):
        captions = [ref['caption'] for ref in references]
        ref_code, pred_code = [[ref['code']] for ref in references], [pred.code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred.rasterize() for pred in predictions]

        scores = [
            crystalbleu.compute(references=ref_code, predictions=pred_code),
            clipscore.compute(references=captions, predictions=pred_image),
            imgscore.compute(references=ref_image, predictions=pred_image),
            kid.compute(references=ref_image, predictions=pred_image),
            eed.compute(references=ref_code, predictions=pred_code),
            compile_success_rate(predictions=predictions)
        ]

        return dict(ChainMap(*scores)) # type: ignore

    return compute

def radar_plot(*args, **kwargs):
    """Make sure that nothing is outside the figure area."""
    fig, ax = plt.subplots(constrained_layout=True)
    ax.axison = False # type: ignore

    return _radar_plot(*args, **kwargs, config={"bbox_to_anchor": None, "legend_loc": "best"}, fig=fig)

if __name__ == "__main__":
    set_seed(0)
    args = parse_args()

    trainset = load_dataset("parquet", data_files=args.trainset, split="train")
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test")

    predictions = dict()
    for model_name, path in map(lambda s: s.split('='), tqdm(args.path, desc="Predicting")):
        (model, tokenizer), captions = load_model(path), testset['caption'] # type: ignore
        generate = TikzGenerator(model=model, tokenizer=tokenizer)

        predictions[model_name] = [generate(caption=c) for c in tqdm(captions, desc=model_name.title(), leave=False)]
        del model, tokenizer, generate

    scores, metrics = dict(), load_metrics(trainset['code'], bs=args.batch_size) # type: ignore
    for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
        scores[model_name] = metrics(references=testset, predictions=prediction)

    if args.output.endswith("json"):
        with open(args.output, "w") as file:
            dump(scores, file)
    else:
        plot = radar_plot(data=list(scores.values()), model_names=list(scores.keys()), invert_range=["KID", "EED"])
        plot.savefig(args.output, bbox_inches = "tight") # type: ignore
