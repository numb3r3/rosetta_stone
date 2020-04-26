import argparse

from rosetta import __version__
from termcolor import colored


def main(args, unknownargs):
    print(args)


def parse_args():
    # create the argument parser
    parser = argparse.ArgumentParser(
        description="%s, a toolkit based on pytorch. "
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

    parser.add_argument("model_name", type=str, required=True, help="the model name")
    parser.add_argument(
        "--yaml",
        type=str,
        default="models.yaml",
        help="the model configuration in yaml file",
    )
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        default="train",
        choices=["train", "eval", "test"],
        help="the running command",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="turn on detailed logging for debug",
    )

    args, unknownargs = parser.parse_known_args()
    return (args, unknownargs)


if __name__ == "__main__":
    args, unknownargs = parse_args()
    main(args, unknownargs)
