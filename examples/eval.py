#!/usr/bin/env python
from argparse import ArgumentParser
from collections import ChainMap, defaultdict
from itertools import pairwise, zip_longest
from json import load as load_json, dump
from os.path import isfile, join
from re import MULTILINE, findall, search
from typing import List, Optional

from datasets import load_dataset
from evaluate.visualization import radar_plot as _radar_plot
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import set_seed

from automatikz.evaluate import load as load_metric
from automatikz.infer import TikzDocument, TikzGenerator, load as load_model

def parse_args():
    argument_parser = ArgumentParser(
        description="Evaluate a range of fine-tuned models."
    )
    argument_parser.add_argument(
        "--cache_dir",
        help="directory where model outputs should be saved to",
    )
    argument_parser.add_argument(
        "--independent_sampling",
        action="store_true",
        help="whether each sample for a prompt was sampled independently or by reusing parts of previous sample",
    )
    argument_parser.add_argument(
        "--use_images",
        action="store_true",
        help="condition the vision encoder on compiled gold documents (only has an effect for clima)",
    )
    argument_parser.add_argument(
        "--batch_size",
        default=1,
        help="batch sizes for metrics that make use of CLIP",
    )
    argument_parser.add_argument(
        "--trainset",
        required=True,
        help="path to the datikz train set (in parquet format)",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
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

def generate_and_repair(generator, **kwargs):
    def parse_errors(log, rootfile=None):
        """
        Returns a dict of (linenr, errormsg) pairs. linenr==0 is a special
        value reserved for errors that do not have a linenumber in rootfile.
        """
        if not rootfile and (match:=search(r"^\((.+)$", log, MULTILINE)):
            rootfile = match.group(1)
        else:
            ValueError("rootfile is unknown!")

        errors = dict()
        for file, line, error in findall(r'^(.+):(\d+):(.+)$', log, MULTILINE):
            if file == rootfile:
                errors[int(line)] = error.strip()
            else: # error occurred in other file
                errors[0] = error.strip()
        #if search(rf"^! {escape(emergency:='Emergency stop.')}$", log, MULTILINE):
        #    errors[0] = emergency

        return errors

    def _generate_and_repair(
        prompt: str,
        try_fix: bool | int = True,
        return_list: bool = False,
        _snippet: str  = "",
        _offset: int = 1,
        _prev_first_error: Optional[int] = None,
        **kwargs,
    ) -> TikzDocument | List[TikzDocument]:
        """
        Internal generate function which tries to fix errors by resampling after the first error.
            prompt: the prompt to generate the tikzpicture
            try_fix: Try to fix errors by resampling or not. If it is an int: how often to attempt resampling.
            return_list: whether to return only the last document or all documents in a list
            kwargs: generation kwargs
        Internal parameters for recursion:
            _offset: how many lines before the first error should be additionally discarded
            _prev_first_error: the first error from the previous sampling attempt
            _retained_text: already generated tikz code to retain
        """
        try:
            tikz = generator(caption=prompt, snippet=_snippet, **kwargs)
        except ValueError:
            if _snippet != "": # input length might exceed max model length
                tikz = generator(caption=prompt, **kwargs)
            else:
                raise
        # If there are errors in the code, try to fix them, unless we have
        # still got an image that isn't empty
        if try_fix and not tikz.has_content:
            errors = parse_errors(tikz.log) or [0]
            # if the first error moves to a different line, reset offset
            if (first_error := min(errors)) != _prev_first_error:
                _offset = 1
            new_tikz = _generate_and_repair(
                prompt=prompt,
                # NOTE: this assumes that \n is it's own token (something which TextStreamer does also)
                _snippet="".join(tikz.code.splitlines(keepends=True)[:max(first_error-_offset, 0)]),
                try_fix=try_fix if isinstance(try_fix, bool) else (try_fix - 1),
                return_list=return_list,
                _offset=max(4*_offset, 2**12), # increase offset (with a maximum limit)
                _prev_first_error=first_error,
                **kwargs
            )
            return [tikz] + new_tikz if return_list else new_tikz # type: ignore
        return [tikz] if return_list else tikz

    return _generate_and_repair(**kwargs)


def load_metrics(trainset, bs=1, independent_sampling=True):
    crystalbleu = load_metric("crystalbleu", corpus=trainset)
    clipscore = load_metric("clipscore", batch_size=bs)
    imgscore = load_metric("clipscore", batch_size=bs, clip_model=clipscore.model, image_to_image=True) # type: ignore
    kid = load_metric("kid", batch_size=bs)
    eed = load_metric("eed")

    def compile_sampling_rate(predictions):
        samples = list()
        for preds in predictions:
            if independent_sampling:
                samples.append(len(preds))
            else:
                samples.append(1) # add first sample
                for p1, p2 in pairwise(preds):
                    for idx, (l1, l2) in enumerate(zip_longest(p1.code.splitlines(), (p2_lines:=p2.code.splitlines()))):
                        if l1 != l2:
                            # for subsequent samples only add fraction of sample
                            # that is new (i.e., resampled code after error)
                            samples.append(len(p2_lines[idx:])/len(p2_lines))
                            break

        return {"CSR": sum(samples)/len(predictions)}

    def compute(references, predictions):
        captions = [ref['caption'] for ref in references]
        ref_code, pred_code = [[ref['code']] for ref in references], [pred[-1].code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred[-1].rasterize() for pred in predictions]
        assert all(pred[-1].has_content for pred in predictions)

        scores = [
            crystalbleu.compute(references=ref_code, predictions=pred_code),
            clipscore.compute(references=captions, predictions=pred_image),
            imgscore.compute(references=ref_image, predictions=pred_image),
            kid.compute(references=ref_image, predictions=pred_image),
            eed.compute(references=ref_code, predictions=pred_code),
            compile_sampling_rate(predictions=predictions)
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
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test").sort("caption")

    predictions = defaultdict(list)
    for model_name, path in map(lambda s: s.split('='), tqdm(args.path, desc="Predicting")):
        if path.endswith("json"):
            with open(path) as f:
                predictions[model_name] = [[TikzDocument(code) for code in sample] for sample in load_json(f)]
        else:
            model, tokenizer = load_model(path)
            generator = TikzGenerator(model=model, tokenizer=tokenizer)
            cache_file = join(args.cache_dir, f'{model_name}.json') if args.cache_dir else None

            if cache_file and isfile(cache_file):
                with open(cache_file) as f:
                    predictions[model_name] = [[TikzDocument(code) for code in sample] for sample in load_json(f)]

            try:
                for idx, item in enumerate(tqdm(testset, desc=model_name.title(), leave=False)): # type: ignore
                    if idx >= len(predictions[model_name]):
                        if args.use_images:
                            tikz = generate_and_repair(generator, prompt=item['caption'], image=item['image'], return_list=True)
                        else:
                            tikz = generate_and_repair(generator, prompt=item['caption'], return_list=True)
                        predictions[model_name].append(tikz)
                del model, tokenizer, generator
            finally:
                if cache_file:
                    with open(cache_file, 'w') as f:
                        dump([[p.code for p in ps] for ps in predictions[model_name]], f)
            #import sys
            #sys.exit()

    scores = dict()
    metrics = load_metrics(trainset['code'], bs=args.batch_size, independent_sampling=args.independent_sampling) # type: ignore
    for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
        scores[model_name] = metrics(references=testset, predictions=prediction)

    if args.output.endswith("json"):
        with open(args.output, "w") as file:
            dump(scores, file)
    else:
        plot = radar_plot(data=list(scores.values()), model_names=list(scores.keys()), invert_range=["KID", "EED", "CSR"])
        plot.savefig(args.output, bbox_inches = "tight") # type: ignore
