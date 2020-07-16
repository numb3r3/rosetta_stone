import os
import time


def _create_optimizer(hparams):
    from ..core import optimizers

    optim = hparams.pop("optimizer")
    if optim == "SGD":
        optimizer = optimizers.SGD(
            lr=hparams["learning_rate"],
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
    from ..helper import load_yaml_params, create_model, create_dataio
    from coolname import generate_slug

    hparams = load_yaml_params(args.yaml_path, args.model_name, cli_args=unused_argv)
    model = create_model(hparams)
    dataio = create_dataio(hparams)

    # setup distributed training env
    if is_distributed() or args.use_horovod:
        init_distributed(use_horovod=args.use_horovod)

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

    # adjust learning rate decay parameter
    from torch.utils.data import DataLoader

    total_size = (
        len(train_loader.dataset)
        if isinstance(train_loader, DataLoader)
        else len(train_loader)
    )
    epoch_steps = total_size // hparams["batch_size"]

    if hparams["lr_warmup_epoch"] > 0:
        hparams["lr_warmup_steps"] = int(hparams["lr_warmup_epoch"] * epoch_steps)
    if hparams["lr_constant_epochs"] > 0:
        hparams["lr_constant_steps"] = int(hparams["lr_constant_epochs"] * epoch_steps)
    if hparams["lr_decay_epoch"] > 0:
        hparams["lr_decay_steps"] = hparams["lr_decay_epoch"] * epoch_steps

    print(hparams)

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

    for epoch in range(hparams["num_epochs"]):
        epoch_start_time = time.time()

        # train for one epoch
        trainer.train(train_loader, **hparams)

        # evaluate on validation set
        eval_metrics = trainer.eval(eval_loader, **hparams)

        # save checkpoint at each epoch
        trainer.save_checkpoint(eval_metrics, **hparams)

        eval_metric_key = hparams["checkpoint_selector"]["eval_metric"]

        report_msg = "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | valid {} {:5.3f}".format(
            trainer.epoch,
            (time.time() - epoch_start_time),
            eval_metrics["loss"],
            eval_metric_key,
            eval_metrics[eval_metric_key],
        )

        for k in eval_metrics.keys():
            if k in ["loss", eval_metric_key]:
                continue
            report_msg += " | {} {:5.3f}".format(k, eval_metrics[k])

        logx.msg("-" * 89)
        logx.msg(report_msg)
        logx.msg("-" * 89)


def eval(args, unused_argv):
    from ..utils.logx import logx
    from ..core import trainers
    from ..helper import load_yaml_params, create_model, create_dataio
    from collections import defaultdict

    logx.rank0 = True
    logx.epoch = defaultdict(lambda: 0)
    logx.no_timestamp = False

    hparams = load_yaml_params(args.yaml_path, args.model_name, cli_args=unused_argv)
    model = create_model(hparams)
    dataio = create_dataio(hparams)

    # data loading code
    eval_data_path = hparams.pop("eval_data_path")
    num_workers = hparams.pop("dataloader_workers")

    eval_loader = dataio.create_data_loader(
        eval_data_path, mode="eval", num_workers=num_workers, **hparams
    )

    optimizer = _create_optimizer(hparams)
    lr_scheduler = _create_lr_scheduler(hparams)
    trainer = trainers.Trainer(
        model, optimizer, lr_scheduler=lr_scheduler, resume=args.resume_from, **hparams
    )

    metric_key = hparams["checkpoint_selector"]["eval_metric"]

    start_time = time.time()

    # evaluate on validation set
    metrics = trainer.eval(eval_loader, **hparams)
    report_msg = "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | valid {} {:5.3f}".format(
        trainer.epoch,
        (time.time() - start_time),
        metrics["loss"],
        metric_key,
        metrics[metric_key],
    )

    for k in metrics.keys():
        if k in ["loss", metric_key]:
            continue
        report_msg += " | {} {:5.3f}".format(k, metrics[k])

    logx.msg("-" * 89)
    logx.msg(report_msg)
    logx.msg("-" * 89)
