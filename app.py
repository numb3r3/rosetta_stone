import argparse
import importlib

from rosetta import __version__, helper
from rosetta.core import lr_schedulers, optimizers, trainers
from termcolor import colored
from torch.nn import functional as F


def run_train(model, data_loader, eval_loader=None):
    optimizer = optimizers.SGD()
    scheduler = lr_schedulers.StepLR(9)
    trainer = trainers.Trainer(model, optimizer, scheduler=scheduler)
    trainer.train(data_loader)


def main(args, unused_argv):

    logger = helper.set_logger("rosetta", verbose=True)

    cli_args = helper.parse_cli_args(unused_argv) if unused_argv else None
    hparams = helper.parse_args("app.yaml", args.model_name, "default")

    if cli_args:
        # useful when changing args for prediction
        logger.info("override args with cli args ...")
        for k, v in cli_args.items():
            if k in hparams and hparams.get(k) != v:
                logger.info("%20s: %20s -> %20s" % (k, hparams.get(k), v))
                hparams[k] = v
            elif k not in hparams:
                logger.warning("%s is not a valid attribute! ignore!" % k)

    logger.info("current parameters")
    for k, v in sorted(hparams.items()):
        if not k.startswith("_"):
            logger.info("%20s = %-20s" % (k, v))

    model_pkg = importlib.import_module(hparams["model_package"])
    model_cls_ = getattr(model_pkg, hparams.get("model_class", "Model"))
    model = model_cls_(**hparams)

    dataio_pkg = importlib.import_module(hparams["dataio_package"])
    dataio_cls_ = getattr(dataio_pkg, hparams.get("dataio_class", "DataIO"))
    dataio = dataio_cls_(**hparams)

    train_loader = dataio.create_data_loader(
        hparams["train_files"], batch_size=hparams["batch_size"], mode="train"
    )

    eval_loader = dataio.create_data_loader(
        hparams["eval_files"], batch_size=hparams["batch_size"], mode="eval"
    )

    run_train(model, train_loader, eval_loader)


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

    parser.add_argument("model_name", type=str, help="the model name")

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

    args, unused_argv = parser.parse_known_args()
    return (args, unused_argv)


if __name__ == "__main__":
    args, unused_argv = parse_args()
    main(args, unused_argv)