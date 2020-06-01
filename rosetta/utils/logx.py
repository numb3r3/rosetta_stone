from runx.logx import logx

from .. import helper


def info(self, msg):
    assert hasattr(self, "logger")
    if not self.rank0:
        return
    self.logger.info(msg)


def warning(self, msg):
    assert hasattr(self, "logger")
    if not self.rank0:
        return
    self.logger.warning(msg)


def debug(self, msg):
    assert hasattr(self, "logger")
    if not self.rank0:
        return
    self.logger.debug(msg)


logx.info = lambda msg: info(logx, msg)
logx.debug = lambda msg: debug(logx, msg)
logx.warning = lambda msg: warning(logx, msg)
