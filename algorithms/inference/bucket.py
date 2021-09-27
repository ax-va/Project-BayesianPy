import math

from pyb4ml.algorithms.inference.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.log_factor import LogFactor


class Bucket:
    def __init__(self, variable):
        self._variable = variable
        self._input_log_factors = []
        self._free_variables = []

    @property
    def input_log_factors(self):
        return self._input_log_factors

    @property
    def variable(self):
        return self._variable

    def add_log_factor(self, log_factor):
        self._input_log_factors.append(log_factor)

    def add_free_variables(self, variables):
        self._free_variables.extend(variables)

    def has_log_factors(self):
        return len(self._input_log_factors) > 0

    def has_free_variables(self):
        return len(set(self._free_variables)) > 0

    def compute_output_log_factor(self):
        # Remove duplicates and sort free variables
        self._set_free_variables()
        # Evaluate free variables
        free_variables_values = FactoredAlgorithm.evaluate_variables(self._free_variables)
        # Compute the function for the output factor
        function_value_dict = {}
        for free_values in free_variables_values:
            free_variables_with_values = tuple(zip(self._free_variables, free_values))
            max_log_factor = ...
            function_value_dict[free_values] = max_log_factor + math.log(
                math.fsum(
                    math.exp(
                        math.fsum(
                            log_factor(
                                *log_factor.filter_variables_with_values(free_variables_with_values),
                                (self._variable, value)
                            ) for log_factor in self._input_log_factors
                        ) - max_log_factor
                    ) for value in self._variable.domain
                )
            )
        return LogFactor(
            variables=self._free_variables,
            function=lambda *values: function_value_dict[values],
            name='log_f_' + self._variable.name
        )

    def _set_free_variables(self):
        # Remove duplicates and sort free variables
        self._free_variables = tuple(sorted(set(self._free_variables), key=lambda x: x.name))