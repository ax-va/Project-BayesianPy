import math

from pyb4ml.modeling.factor_graph.factor import Factor


def log(f):
    def lof_f(x):
        return math.log(f(x))
    return lof_f


class LogFactor(Factor):
    def __init__(self, factor=None, variables=None, function=None, name=None):
        if factor is not None:
            Factor.__init__(self, factor.variables, log(factor.function), 'log_' + factor.name)
        elif variables is not None and function is not None:
            Factor.__init__(self, variables, function, name)
        else:
            raise ValueError('not all sufficient arguments specified')