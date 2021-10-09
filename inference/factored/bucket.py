import math

from pyb4ml.inference.factored.abstract_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.log_factor import LogFactor


class Bucket:
    def __init__(self, variable):
        self._variable = variable
        self._input_log_factors = []
        self._free_variables = []

    @property
    def free_variables(self):
        return self._free_variables

    @property
    def input_log_factors(self):
        return self._input_log_factors

    @property
    def variable(self):
        return self._variable

    def add_log_factor(self, log_factor):
        self._input_log_factors.append(log_factor)

    def has_log_factors(self):
        return len(self._input_log_factors) > 0

    def has_free_variables(self):
        return len(self._free_variables) > 0

    def compute_output_log_factor(self):
        # Evaluate free variables
        free_variables_values = FactoredAlgorithm.evaluate_variables(self._free_variables)
        # Compute the function for the output factor
        function_value_dict = {}
        for free_values in free_variables_values:
            # Zip the free variables with their values
            free_variables_with_values = tuple(zip(self._free_variables, free_values))
            # Save the values of the log-factors depending on the values of the summing variable
            input_log_factors = {
                value:
                    tuple(
                        log_factor(
                            *log_factor.filter_values(*free_variables_with_values), (self._variable, value)
                        ) for log_factor in self._input_log_factors
                    ) for value in self._variable.domain
                }
            # Get the maximum log-factor for computational stability
            max_log_factor = max(max(input_log_factors[value]) for value in self._variable.domain)
            # Save the values of the function for the output log-factor
            function_value_dict[free_values] = max_log_factor + math.log(
                math.fsum(
                    math.exp(
                        math.fsum(input_log_factors[value]) - max_log_factor
                    ) for value in self._variable.domain
                )
            )
        return LogFactor(
            variables=self._free_variables,
            function=lambda *values: function_value_dict[values],
            name='log_f_' + self._variable.name
        )

    def set_free_variables(self):
        self._free_variables = [var
                                for log_factor in self._input_log_factors
                                for var in log_factor.variables
                                if var is not self._variable]
        self._free_variables = tuple(sorted(set(self._free_variables), key=lambda x: x.name))