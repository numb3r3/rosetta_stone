import torch

from .logx import logx


def restore_optimizer(checkpoint, optimizer, scheduler):
    """Check if continuing training from a checkpoint."""
    # if (self.args.model_path is not None
    #         and os.path.isfile(os.path.join(self.args.model_path, "optimizer.pt"))
    #         and os.path.isfile(os.path.join(self.args.model_path, "scheduler.pt"))
    # ):
    #     self.logger.info("Load in optimizer and scheduler states from %s", self.args.model_path)
    #     optimizer.load_state_dict(torch.load(os.path.join(self.args.model_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(self.args.model_path, "scheduler.pt")))
    #     if os.path.isfile(os.path.join(self.args.model_path, "state.bin")):
    #         state = torch.load(os.path.join(self.args.model_path, "state.bin"))
    #         if self.model_checkpoint and hasattr(state, 'best'):
    #             self.model_checkpoint.best = state['best']
    # return optimizer, scheduler
    pass


def load_checkpoint(checkpoint_path, model=None, optimizer=None, amp=None):
    """Load checkpoint.

    Args:
        checkpoint_path (str): path to the saved model (model..epoch-*)
        model (torch.nn.Module):
        optimizer (LRScheduler): optimizer wrapped by LRScheduler class
        amp ():
    Returns:
        topk_list (list): list of (epoch, metric)
    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError('There is no checkpoint')

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError('No checkpoint found at %s' % checkpoint_path)

    # Restore parameters
    if 'avg' not in checkpoint_path:
        epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1
        logx.msg('=> Loading checkpoint (epoch:%d): %s' %
                 (epoch + 1, checkpoint_path))
    else:
        logx.msg('=> Loading checkpoint: %s' % checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # NOTE: fix this later
        optimizer.optimizer.param_groups[0]['params'] = []
        for param_group in list(model.parameters()):
            optimizer.optimizer.param_groups[0]['params'].append(param_group)
    else:
        logger.warning('Optimizer is not loaded.')

    # Restore apex
    if amp is not None:
        amp.load_state_dict(checkpoint['amp_state_dict'])
    else:
        logger.warning('amp is not loaded.')

    if ('optimizer_state_dict' in checkpoint.keys()
            and 'topk_list' in checkpoint['optimizer_state_dict'].keys()):
        topk_list = checkpoint['optimizer_state_dict']['topk_list']
    else:
        topk_list = []
    return topk_list
