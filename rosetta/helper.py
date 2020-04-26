import collections
import logging
from typing import List

from ruamel.yaml import YAML


def get_logger(name: str, log_format) -> logging.Logger:
    import logging.handlers

    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(log_format, datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    return logger


def set_logger(name: str, log_format: str = None, verbose: bool = False):
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        if not log_format:
            log_format = (
                "%(levelname)-.1s:"
                + name
                + ":[%(filename).8s:%(funcName).8s:%(lineno)3d]:%(message)s"
            )
        formatter = logging.Formatter(log_format, datefmt="%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.handlers = []
        logger.addHandler(console_handler)

    return logger


def parse_cli_args(args: List[str]):
    cli_args = collections.defaultdict(list)
    if args:
        for k, v in ((k.lstrip("-"), v) for k, v in (a.split("=") for a in args)):
            cli_args[k].append(v)

        for k, v in cli_args.items():
            parsed_v = [s for s in (parse_arg(vv) for vv in v) if s is not None]
            if len(parsed_v) > 1:
                cli_args[k] = parsed_v
            if len(parsed_v) == 1:
                cli_args[k] = parsed_v[0]
    return cli_args


def parse_arg(v: str):
    if v.startswith("[") and v.endswith("]"):
        # function args must be immutable tuples not list
        tmp = v.replace("[", "").replace("]", "").strip().split(",")
        if len(tmp) > 0:
            return [parse_arg(_.strip()) for _ in tmp]
        else:
            return []
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            if len(v) == 0:
                # ignore it when the parameter is empty
                v = None
            elif v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
    return v


def parse_args(yaml_path: str, model_name: str, default_set=None):
    with open("default.yaml") as fp:
        configs = YAML().load(fp)
        default_cfg = configs[default_set]

        hparams = default_cfg.copy()
        hparams["model_name"] = model_name

        if yaml_path:
            with open(yaml_path) as fp:
                customized = YAML().load(fp)
                model_params = customized[model_name]

                hparams.update(model_params)

    return hparams
