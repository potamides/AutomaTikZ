from re import sub

from svg2tikz.extensions.tikz_export import convert_svg

from . import get_creation_time, load_silent

REPO = "evanthebouncy/chatgpt-svg"
CREATED = get_creation_time(REPO)

def convert(svg):
    tikz = convert_svg(svg, wrap=True, crop=True)

    # https://github.com/xyz2tex/svg2tikz/issues/30
    tikz = sub(r'\.?0{2}\d*', "", tikz)
    tikz = sub(4 * '\n', "\n", tikz).strip()

    return tikz

def load(tsv):
    dataset = load_silent("csv", sep="\t", data_files=tsv, split="train")
    for idx, item in enumerate(dataset, 1):
        caption = item['prompt'].removeprefix("Using the SVG format, output ").split(".")[0] + "."
        tikz = convert(item['svg'])

        yield {
            "caption": caption,
            "code": tikz,
            "date": CREATED,
            "uri": f"https://github.com/{REPO}/blob/master/data.tsv#L{idx}"
        }
