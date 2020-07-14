import collections
import logging
import os
import sys
from typing import List

from ruamel.yaml import YAML


def get_logger(name: str, log_format: str = None) -> logging.Logger:
    import logging.handlers

    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    if not log_format:
        log_format = (
            "%(levelname)-.1s:"
            + name
            + ":[%(filename).8s:%(funcName).8s:%(lineno)3d]:%(message)s"
        )
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
            if len(parsed_v) == 0:
                cli_args[k] = None
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


def parse_args(
    yaml_path: str,
    model_name: str,
    default_set: str = None,
    default_yaml_file: str = None,
):
    with open(default_yaml_file or "default.yaml") as fp:
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


def load_yaml_params(yaml_path: str, model_name: str, cli_args=None):
    from pkg_resources import resource_filename

    cli_args = parse_cli_args(cli_args) if cli_args else None
    default_yaml_file = resource_filename(
        "rosetta", "/".join(("resources", "default.yaml"))
    )
    hparams = parse_args(yaml_path, model_name, "default", default_yaml_file)

    if cli_args:
        # useful when changing params defined in YAML
        print("override parameters with cli args ...")
        for k, v in cli_args.items():
            if k in hparams and hparams.get(k) != v:
                print("%20s: %20s -> %20s" % (k, hparams.get(k), v))
                hparams[k] = v
            elif k not in hparams:
                print("%s is not a valid attribute! ignore!" % k)

    print("current parameters")
    for k, v in sorted(hparams.items()):
        if not k.startswith("_"):
            print("%20s = %-20s" % (k, v))
    return hparams


def create_model(hparams, resume_from: str=None):
    from .utils.pathlib import import_path

    model_pkg_name, model_cls_name = hparams["model_module"].split(":")
    model_pkg_path = os.path.join(*model_pkg_name.split(".")) + ".py"
    model_pkg = import_path(model_pkg_path)
    model_cls_ = getattr(model_pkg, model_cls_name)
    model = model_cls_(**hparams)

    # model_pkg_name, model_cls_name = hparams["model_module"].split(':')
    # model_pkg = importlib.import_module(model_pkg_name)
    # model_cls_ = getattr(model_pkg, model_cls_name)
    # model = model_cls_(**hparams)

    if resume_from:
        if os.path.isfile(resume_from):
            print("=> loading checkpoint '{}'".format(resume_from))
            checkpoint = torch.load(resume_from, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(resume_from))
            sys.exit(1)

    return model


def create_dataio(hparams):
    from .utils.pathlib import import_path

    dataio_pkg_name, dataio_cls_name = hparams["dataio_module"].split(":")
    dataio_pkg_path = os.path.join(*dataio_pkg_name.split(".")) + ".py"
    dataio_pkg = import_path(dataio_pkg_path)
    dataio_cls_ = getattr(dataio_pkg, dataio_cls_name)
    dataio = dataio_cls_(**hparams)

    return dataio


class PathImporter:
    @staticmethod
    def _get_module_name(absolute_path):
        module_name = os.path.basename(absolute_path)
        module_name = module_name.replace(".py", "")
        return module_name

    @staticmethod
    def add_modules(*paths):
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "cannot import module from %s, file not exist", p
                )
            module, spec = PathImporter._path_import(p)
        return module

    @staticmethod
    def _path_import(absolute_path):
        import importlib.util

        module_name = PathImporter._get_module_name(absolute_path)
        spec = importlib.util.spec_from_file_location(module_name, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
        return module, spec


default_logger = get_logger("rosetta")  #: a logger at the global-level
