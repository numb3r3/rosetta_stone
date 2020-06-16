import argparse
from datetime import datetime
import importlib
import os
import time
from typing import Dict, Iterable

from rosetta import __version__, helper
from rosetta.core import lr_schedulers, optimizers, trainers
from rosetta.utils.distribute import get_world_size, get_global_rank, init_distributed, is_distributed
from rosetta.utils.logx import logx
from termcolor import colored
from torch.utils.data import DataLoader


def run_train(
    model,
    data_loader: Iterable or DataLoader,
    eval_loader: Iterable or DataLoader = None,
    use_horovod: bool = False,
    use_amp: bool = False,
    resume: str = None,
    hparams: Dict = {},
):
    optim = hparams.pop("optimizer")
    if optim == "SGD":
        optimizer = optimizers.SGD(
            lr=hparams["learning_rate"] * get_world_size(),
            weight_decay=hparams["weight_decay_rate"],
            momentum=hparams.get("momentum", 0),
            dampening=hparams.get("dampening", 0),
        )
    else:
        optimizer = {"Adam": optimizers.Adam, "AdamW": optimizers.AdamW}.get(optim)(
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay_rate"],
            betas=(hparams.get("adam_beta1", 0.9), hparams.get("adam_beta2", 0.999)),
        )

    lr_scheduler = lr_schedulers.DecayedLRWithWarmup(
        warmup_steps=hparams["lr_warmup_steps"],
        constant_steps=hparams["lr_constant_steps"],
        decay_method=hparams["lr_decay_method"],
        decay_steps=hparams["lr_decay_steps"],
        decay_rate=hparams["lr_decay_rate"],
    )
    trainer = trainers.Trainer(
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        use_horovod=use_horovod,
        use_amp=use_amp,
        resume=resume,
        **hparams,
    )

    for epoch in range(hparams["num_epochs"]):
        epoch_start_time = time.time()

        # train for one epoch
        trainer.train(data_loader, **hparams)

        # evaluate on validation set
        eval_metrics = trainer.eval(eval_loader, **hparams)

        # save checkpoint at each epoch
        trainer.save_checkpoint(eval_metrics, **hparams)

        logx.msg("-" * 89)
        logx.msg(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f}".format(
                epoch, (time.time() - epoch_start_time), eval_metrics["loss"]
            )
        )
        logx.msg("-" * 89)


def main(args, unused_argv):

    logger = helper.set_logger("rosetta", verbose=True)

    cli_args = helper.parse_cli_args(unused_argv) if unused_argv else None
    hparams = helper.parse_args("app.yaml", args.model_name, "default")

    if is_distributed() or args.use_horovod:
        init_distributed(use_horovod=args.use_horovod)

    # hacks: get_global_rank() would return -1 for standalone training
    global_rank = max(0, get_global_rank())

    log_dir = hparams.get(
        "log_dir", os.path.join(hparams["log_dir_prefix"], args.model_name)
    )

    # from coolname import generate_slug

    args_str = "use_amp-%d-use_horovod-%d" % (args.use_amp, args.use_horovod)

    log_name = args_str + "-" + datetime.now().strftime("%Y-%m-%d")
    logx.initialize(
        logdir=os.path.join(log_dir, log_name),
        coolname=True,
        tensorboard=True,
        global_rank=global_rank,
        eager_flush=True,
        hparams=hparams,
    )

    # attach logger to logx
    logx.logger = logger

    if cli_args:
        # useful when changing params defined in YAML
        # logger.info("override parameters with cli args ...")
        logx.msg("override parameters with cli args ...")
        for k, v in cli_args.items():
            if k in hparams and hparams.get(k) != v:
                # logger.info("%20s: %20s -> %20s" % (k, hparams.get(k), v))
                logx.msg("%20s: %20s -> %20s" % (k, hparams.get(k), v))
                hparams[k] = v
            elif k not in hparams:
                # logger.warning("%s is not a valid attribute! ignore!" % k)
                logx.msg("%s is not a valid attribute! ignore!" % k)

    # logger.info("current parameters")
    logx.msg("current parameters")
    for k, v in sorted(hparams.items()):
        if not k.startswith("_"):
            # logger.info("%20s = %-20s" % (k, v))
            logx.msg("%20s = %-20s" % (k, v))

    model_pkg = importlib.import_module(hparams["model_package"])
    model_cls_ = getattr(model_pkg, hparams.get("model_class", "Model"))
    model = model_cls_(**hparams)

    dataio_pkg = importlib.import_module(hparams["dataio_package"])
    dataio_cls_ = getattr(dataio_pkg, hparams.get("dataio_class", "DataIO"))
    dataio = dataio_cls_(**hparams)

    # Data loading code
    train_data_path = hparams.pop("train_data_path")
    eval_data_path = hparams.pop("eval_data_path")
    num_workers = hparams.pop("dataloader_workers")
    batch_size = hparams.pop("batch_size")

    train_loader = dataio.create_data_loader(
        train_data_path,
        batch_size=batch_size,
        mode="train",
        num_workers=num_workers,
        **hparams,
    )

    eval_loader = dataio.create_data_loader(
        eval_data_path,
        batch_size=batch_size,
        mode="eval",
        num_workers=num_workers,
        **hparams,
    )

    run_train(
        model,
        train_loader,
        eval_loader,
        use_horovod=args.use_horovod,
        use_amp=args.use_amp,
        resume=args.resume,
        hparams=hparams,
    )


def parse_args():
    # create the argument parser
    parser = argparse.ArgumentParser(
        description="%s, a toolkit based on pytorch. "
        "Visit %s for tutorials and documents."
        % (
            colored("rosetta stone v%s" % __version__, "green"),
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
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="use apex for automatic mixed precision training",
    )

    parser.add_argument(
        "--use_horovod",
        action="store_true",
        default=False,
        help="use horovod for distributed training",
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
