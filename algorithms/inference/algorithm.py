import copy

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.variable import Variable
from pyb4ml.models.factor_graphs.model import Model


class InferenceAlgorithm:
    def __init__(self, model: Model):
        # Save the factorization object
        self._model = model
        # Encapsulate the factors and variables inside the algorithm
        # self._factors and self._variables created
        self._encapsulate_factors_and_variables()
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

    def _create_factor_variables(self, model_factor):
        factor_variables = []
        for model_factor_variable in model_factor.variables:
            index = self._model.variables.index(model_factor_variable)
            factor_variables.append(self._variables[index])
        return tuple(factor_variables)

    def _encapsulate_factors_and_variables(self):
        # Encapsulate the factors and variables inside the algorithm
        # Deeply copy the variables
        self._variables = tuple(copy.deepcopy(self._model.variables))
        # Unlink the factors from the variables
        for variable in self._variables:
            variable.unlink_factors()
        # Create new factors
        self._factors = tuple(
            Factor(
                variables=self._create_factor_variables(model_factor),
                function=copy.deepcopy(model_factor.function),
                name=copy.deepcopy(model_factor.name)
            ) for model_factor in self._model.factors
        )

    def set_query(self, query: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        if query not in self._model.variables:
            raise ValueError('there is no variable in the factorization that corresponds to the query')
        # Make sure that the query variable is from the encapsulated sequence in this algorithm
        self._query = self._variables[self._model.variables.index(query)]