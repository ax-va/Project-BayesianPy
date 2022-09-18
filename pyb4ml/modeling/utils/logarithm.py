import math


def logarithm(func):
    def logarithm_func(*t):
        return math.log(func(*t))
    return logarithm_func
