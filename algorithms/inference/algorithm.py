import copy

from pyb4ml.modeling.factor_graph.factorization import Factorization
from pyb4ml.modeling.factor_graph.variable import Variable


class InferenceAlgorithm:
    def __init__(self, factorization: Factorization):
        # Save the factorization object
        self._factorization = factorization
        # Copy deeply the factors to encapsulate them inside the algorithm
        self._factors = copy.deepcopy(self._factorization.factors)
        # Copy deeply the variables to encapsulate them inside the algorithm
        self._variables = copy.deepcopy(self._factorization.variables)
        # Query is not yet set
        self._query = None
        # Evidence is not given
        self._evidence = None
        # Probability distribution P of interest
        self._distribution = None

    @property
    def factor_leaves(self):
        return tuple(factor for factor in self._factors if factor.is_leaf())

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())

    def set_query(self, query: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        if query not in self._factorization.variables:
            raise ValueError('there is no variable in the factorization that corresponds to the query')
        # Make sure that the query variable is from the encapsulated sequence in this algorithm
        self._query = self._variables[self._factorization.variables.index(query)]