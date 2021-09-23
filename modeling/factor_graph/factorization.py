import numpy as np


class Factorization:
    def __init__(self, factors, variables):
        self._factors = tuple(sorted(set(factors), key=lambda f: f.name))
        self._variables = tuple(sorted(set(variables), key=lambda v: v.name))
        self._factorization = {factor.variables: factor for factor in self._factors}

    @property
    def factorization(self):
        return self._factorization

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables
