from runx.logx import logx

from .. import helper


def info(self, msg, logger=None):
    if not self.rank0:
        return
    if logger is None:
        logger = helper.get_logger(__name__)

    logger.info(msg)


logx.info = info
