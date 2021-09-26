import math

from pyb4ml.algorithms.inference.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor import Factor


class Bucket:
    def __init__(self, variable):
        self._variable = variable
        self._factors = []
        self._free_variables = []

    @property
    def factors(self):
        return self._factors

    @property
    def variable(self):
        return self._variable

    def add_factor(self, factor):
        self._factors.append(factor)

    def add_free_variables(self, variables):
        self._free_variables.extend(variables)

    def has_factors(self):
        return len(self._factors) > 0

    def has_free_variables(self):
        return len(set(self._free_variables)) > 0

    def compute_factor_with_free_variables(self):
        # Remove duplicates and sort free variables
        self._set_free_variables()
        # Evaluate free variables
        evaluated_free_variables = FactoredAlgorithm.evaluate_variables(self._free_variables)
        function_value_dict = {}
        for free_values in evaluated_free_variables:
            free_variables_with_values = tuple(zip(self._free_variables, free_values))
            factor_value = math.fsum(
                math.prod(
                    factor(
                        *factor.filter_variables_with_values(free_variables_with_values),
                        (self._variable, value)
                    ) for factor in self._factors
                ) for value in self._variable.domain
            )
            function_value_dict[free_values] = factor_value
        factor_with_free_variables = Factor(
            variables=self._free_variables,
            function=lambda *values: function_value_dict[values],
            name=f'f_{self._variable}'
        )
        return factor_with_free_variables

    def _set_free_variables(self):
        # Remove duplicates and sort free variables
        self._free_variables = tuple(sorted(set(self._free_variables), key=lambda x: x.name))


