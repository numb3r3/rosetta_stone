import argparse

from termcolor import colored


def add_arg_group(parser, title):
    return parser.add_argument_group(title)


def set_base_parser():
    from .. import __version__

    parser = argparse.ArgumentParser(
        epilog='%s, a toolkit based on pytorch. '
        'Visit %s for tutorials and documents.' % (
            colored('rosetta stone v%s' % __version__, 'green'),
            colored(
                'https://git.huya.com/wangfeng2/rosetta_stone',
                'cyan',
                attrs=['underline'],
            ),
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Rosetta Stone Line Interface',
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help='show Rosetta version',
    )

    return parser


def set_trainer_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'trainer arguments')

    gp.add_argument('model_name', type=str, help='the model name')
    gp.add_argument(
        '--yaml-path',
        type=str,
        default='app.yaml',
        help='a yaml file configs models')

    gp.add_argument(
        '--checkpoint',
        default='',
        type=str,
        metavar='PATH',
        help='path to the checkpoint (default: none)',
    )

    gp.add_argument(
        '--resume-optimizer',
        action='store_true',
        default=False,
        help='resume optimizer (and scheduler) from checkpoint',
    )

    gp.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    gp.add_argument(
        '--use-amp',
        action='store_true',
        default=False,
        help='use apex for automatic mixed precision training',
    )

    gp.add_argument(
        '--use-horovod',
        action='store_true',
        default=False,
        help='use horovod for distributed training',
    )

    gp.add_argument(
        '--use-sync-bn',
        action='store_true',
        default=False,
        help=
        'convert BatchNorm layer to SyncBatchNorm before wrapping Network with DDP',
    )

    gp.add_argument(
        '--use-prefetcher',
        action='store_true',
        default=False,
        help='use prefetcher to speed up data loader',
    )

    return parser


def set_evaluator_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'evaluator arguments')

    gp.add_argument('model_name', type=str, help='the model name')
    gp.add_argument(
        '--yaml-path',
        type=str,
        default='app.yaml',
        help='a yaml file configs models')

    gp.add_argument(
        '--checkpoint',
        default='',
        type=str,
        metavar='PATH',
        help='path to the checkpoint (default: none)',
    )

    gp.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    gp.add_argument(
        '--use-prefetcher',
        action='store_true',
        default=False,
        help='use prefetcher to speed up data loader',
    )

    return parser


def get_main_parser():
    # create the top-level parser
    parser = set_base_parser()
    import os

    sp = parser.add_subparsers(
        dest='cli',
        description='use "%(prog)-8s [sub-command] --help" '
        'to get detailed information about each sub-command',
    )

    set_trainer_parser(
        sp.add_parser(
            'train',
            help='👋 train a pytorch model',
            description='Start to train a pytorch model, '
            'without any extra codes.',
        ))

    set_evaluator_parser(
        sp.add_parser(
            'eval',
            help='👋 evaluate a pytorch model',
            description='Start to eval a pytorch model, '
            'without any extra codes.',
        ))

    return parser
