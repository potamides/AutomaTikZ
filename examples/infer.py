#!/usr/bin/env python
from argparse import ArgumentParser
from sys import flags

from PIL import UnidentifiedImageError
from transformers import set_seed

from automatikz.infer import TikzGenerator, load

try:
    import readline # patches input()
except:
    pass #readline not available

def parse_args():
    argument_parser = ArgumentParser(
        description="Inference helper for fine-tuned models."
    )
    argument_parser.add_argument(
        "--path",
        required=True,
        help="path or url to projector weights or directory where to find models/adapters (local or hub)",
    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    set_seed(0)
    generate = TikzGenerator(*load(parse_args().path), stream=True)
    desc = dict(
        caption="the caption",
        image="optional input fed into CLIP, defaults to the caption (can be a Pillow Image, a URI to an image, or a caption)"
    )

    if flags.interactive:
        print("generate(*args, **kwargs):", str(TikzGenerator.generate.__doc__).strip())
    else:
        print("Starting a REPL for generating TikZ. Arguments:", *[f"\t{k}: {v}" for k, v in desc.items()], sep="\n")
        while True:
            try:
                caption = input("Caption: ")
                image = input("Image (optional): ") if generate.is_multimodal else None
            except (KeyboardInterrupt, EOFError):
                break
            try:
                generate(caption=caption, image=image or None)
            except KeyboardInterrupt:
                pass
            except UnidentifiedImageError:
                print("Error: Cannot identify image file!")
