import numpy as np


class Factorization:
    def __init__(self, factors, variables):
        self._factors = factors
        self._variables = variables

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables
