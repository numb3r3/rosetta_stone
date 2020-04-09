import argparse

from rosetta import __version__
from termcolor import colored


def main(args):
    print(args)


def parse_args():
    # create the commend argument parser
    parser = argparse.ArgumentParser(
        description="%s, a nlp toolkit based on pytorch. "
        "Visit %s for tutorials and documents."
        % (
            colored("Rosetta v%s" % __version__, "green"),
            colored(
                "https://git.huya.com/wangfeng2/rosetta_stone",
                "cyan",
                attrs=["underline"],
            ),
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-m", "--model", type=str, required=True, help="the model name")
    parser.add_argument(
        "-e",
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="the running mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="turn on detailed logging for debug",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
