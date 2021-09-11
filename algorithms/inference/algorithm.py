import copy

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.factorization import Factorization
from pyb4ml.modeling.factor_graph.variable import Variable


class InferenceAlgorithm:
    def __init__(self, factorization: Factorization):
        # Save the factorization object
        self._factorization = factorization
        # Encapsulate the factors and variables inside the algorithm
        # self._factors and self._variables created
        self._encapsulate_factors_and_variables()
        # Query is not yet set
        self._query = None
        # Evidence is not given
        self._evidence = None
        # Probability distribution P of interest
        self._distribution = None

    @staticmethod
    def create_factor_variables(old_factor, old_variables, new_variables):
        new_factor_variables = []
        for old_variable in old_factor.variables:
            index = old_variables.index(old_variable)
            new_factor_variables.append(new_variables[index])
        return tuple(new_factor_variables)

    @property
    def factor_leaves(self):
        return tuple(factor for factor in self._factors if factor.is_leaf())

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())

    def _encapsulate_factors_and_variables(self):
        # Encapsulate the factors and variables inside the algorithm
        # Deeply copy the variables
        self._variables = tuple(copy.deepcopy(self._factorization.variables))
        # Unlink the factors from the variables
        for variable in self._variables:
            variable.unlink_factors()
        # Create new factors
        self._factors = tuple(
            Factor(
                variables=self.create_factor_variables(
                    old_factor=factor,
                    old_variables=self._factorization.variables,
                    new_variables=self._variables
                ),
                function=copy.deepcopy(factor.function),
                name=copy.deepcopy(factor.name)
            ) for factor in self._factorization.factors
        )

    def set_query(self, query: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        if query not in self._factorization.variables:
            raise ValueError('there is no variable in the factorization that corresponds to the query')
        # Make sure that the query variable is from the encapsulated sequence in this algorithm
        self._query = self._variables[self._factorization.variables.index(query)]