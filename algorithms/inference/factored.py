import copy

from bayesian.modeling.factor_graph.factorization import Factorization


class Factored:
    def __init__(self, factorization: Factorization):
        # Save the factorization object
        self._factorization = factorization
        # Copy deeply the factors to encapsulate them inside the algorithm
        self._factors = copy.deepcopy(self._factorization.factors)
        # Copy deeply the variables to encapsulate them inside the algorithm
        self._variables = copy.deepcopy(self._factorization.variables)

    @property
    def factor_leaves(self):
        return tuple(factor for factor in self._factors if factor.is_leaf())

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())