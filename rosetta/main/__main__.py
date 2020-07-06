import sys


__all__ = ["main"]


def _get_run_args(print_args: bool = True):
    from termcolor import colored

    from .parser import get_main_parser

    parser = get_main_parser()
    if len(sys.argv) > 1:
        from argparse import _StoreAction, _StoreTrueAction

        # args = parser.parse_args()
        args, unused_argv = parser.parse_known_args()

        return args, unused_argv
    else:
        parser.print_help()
        exit()


def main():
    """The main entrypoint of the CLI """
    from . import api

    args, unused_argv = _get_run_args()

    getattr(api, args.cli.replace("-", "_"))(args, unused_argv)


if __name__ == "__main__":
    main()
