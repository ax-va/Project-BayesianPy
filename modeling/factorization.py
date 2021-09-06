import numpy as np


class Factorization:
    def __init__(self, factors, variables):
        self._factors = factors
        self._variables = variables

    @staticmethod
    def get_factor_leaves(factors):
        return tuple(factor for factor in factors if factor.is_leaf())

    @staticmethod
    def get_variable_leaves(variables):
        return tuple(variable for variable in variables if variable.is_non_isolated_leaf())

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables
