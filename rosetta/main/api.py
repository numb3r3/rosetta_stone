import importlib
import os
import sys
import time


def _parse_hparams(args, unused_argv):
    from .. import helper
    from pkg_resources import resource_filename

    cli_args = helper.parse_cli_args(unused_argv) if unused_argv else None
    default_yaml_file = resource_filename(
        "rosetta", "/".join(("resources", "default.yaml"))
    )
    hparams = helper.parse_args(
        args.yaml_path, args.model_name, "default", default_yaml_file
    )

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


def _create_model(hparams):
    from ..utils.pathlib import import_path

    model_pkg_name, model_cls_name = hparams["model_module"].split(":")
    model_pkg_path = os.path.join(*model_pkg_name.split(".")) + ".py"
    model_pkg = import_path(model_pkg_path)
    model_cls_ = getattr(model_pkg, model_cls_name)
    model = model_cls_(**hparams)

    # model_pkg_name, model_cls_name = hparams["model_module"].split(':')
    # model_pkg = importlib.import_module(model_pkg_name)
    # model_cls_ = getattr(model_pkg, model_cls_name)
    # model = model_cls_(**hparams)

    return model


def _create_dataio(hparams):
    from ..utils.pathlib import import_path

    dataio_pkg_name, dataio_cls_name = hparams["dataio_module"].split(":")
    dataio_pkg_path = os.path.join(*dataio_pkg_name.split(".")) + ".py"
    dataio_pkg = import_path(dataio_pkg_path)
    dataio_cls_ = getattr(dataio_pkg, dataio_cls_name)
    dataio = dataio_cls_(**hparams)

    return dataio


def _create_optimizer(hparams):
    from ..core import optimizers

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
    return optimizer


def _create_lr_scheduler(hparams):
    from ..core import lr_schedulers

    lr_scheduler = lr_schedulers.DecayedLRWithWarmup(
        warmup_steps=hparams["lr_warmup_steps"],
        constant_steps=hparams["lr_constant_steps"],
        decay_method=hparams["lr_decay_method"],
        decay_steps=hparams["lr_decay_steps"],
        decay_rate=hparams["lr_decay_rate"],
    )
    return lr_scheduler


def train(args, unused_argv):
    from ..utils.distribute import (
        get_global_rank,
        get_world_size,
        init_distributed,
        is_distributed,
    )
    from ..utils.logx import logx
    from ..core import trainers
    from coolname import generate_slug

    hparams = _parse_hparams(args, unused_argv)
    model = _create_model(hparams)
    dataio = _create_dataio(hparams)

    # setup distributed training env
    if is_distributed() or args.use_horovod:
        init_distributed(use_horovod=args.use_horovod)

    # data loading code
    train_data_path = hparams.pop("train_data_path")
    eval_data_path = hparams.pop("eval_data_path")
    num_workers = hparams.pop("dataloader_workers")

    train_loader = dataio.create_data_loader(
        train_data_path, mode="train", num_workers=num_workers, **hparams
    )
    eval_loader = dataio.create_data_loader(
        eval_data_path, mode="eval", num_workers=num_workers, **hparams
    )

    optimizer = _create_optimizer(hparams)
    lr_scheduler = _create_lr_scheduler(hparams)
    trainer = trainers.Trainer(
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        use_horovod=args.use_horovod,
        use_amp=args.use_amp,
        resume=args.resume_from,
        **hparams,
    )

    # NOTE: get_global_rank() would return -1 for standalone training
    global_rank = max(0, get_global_rank())

    log_dir = hparams.get(
        "log_dir", os.path.join(hparams["log_dir_prefix"], args.model_name)
    )
    suffix_model_id = hparams["suffix_model_id"]
    log_name = suffix_model_id + ("-" if suffix_model_id else "") + generate_slug(2)

    logx.initialize(
        logdir=os.path.join(log_dir, log_name),
        coolname=True,
        tensorboard=True,
        global_rank=global_rank,
        eager_flush=True,
        hparams=hparams,
    )

    for epoch in range(hparams["num_epochs"]):
        epoch_start_time = time.time()

        # train for one epoch
        trainer.train(train_loader, **hparams)

        # evaluate on validation set
        eval_metrics = trainer.eval(eval_loader, **hparams)

        # save checkpoint at each epoch
        trainer.save_checkpoint(eval_metrics, **hparams)

        eval_metric_key = hparams["checkpoint_selector"]["eval_metric"]

        logx.msg("-" * 89)
        logx.msg(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | valid metric {} {:5.3f}".format(
                epoch,
                (time.time() - epoch_start_time),
                eval_metrics["loss"],
                eval_metric_key,
                eval_metrics[eval_metric_key],
            )
        )
        logx.msg("-" * 89)
