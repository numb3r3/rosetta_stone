from runx.logx import logx


def to_np(x):
    """https://github.com/yunjey/pytorch-
    tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20.

    :param x:
    :return:
    """
    return x.data.cpu().numpy()


def info(self, msg):
    assert hasattr(self, 'logger')
    if not self.rank0:
        return
    self.logger.info(msg)


def warning(self, msg):
    assert hasattr(self, 'logger')
    if not self.rank0:
        return
    self.logger.warning(msg)


def debug(self, msg):
    assert hasattr(self, 'logger')
    if not self.rank0:
        return
    self.logger.debug(msg)


logx.info = lambda msg: info(logx, msg)
logx.debug = lambda msg: debug(logx, msg)
logx.warning = lambda msg: warning(logx, msg)


def summary_model(self, model):
    import numpy as np

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameter = {}'.format(total_params))


logx.summary_model = lambda m: summary_model(logx, m)
