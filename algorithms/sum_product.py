import math

from bayesian.algorithms.factored import Factored
from bayesian.modeling.factorization import Factorization
from bayesian.modeling.variable import Variable


class SumProduct(Factored):
    def __init__(self, factorization: Factorization):
        Factored.__init__(self, factorization)
        # Query is not yet set
        self._query_variable = None
        # Evidence is not given
        self._evidence = None

    def run(self):
        # Startup
        # Set the evidence
        # Can you get an immediate response to the query?
        self._initialize()
        # Running the main loop
        while self._running:
            for factor in self._next_factors:
                pass
            for variable in self._next_variables:
                pass

    def set_query(self, query: Variable):
        # Single variable 'query' of interest for computing P(query) or P(query|evidence)
        self._query_variable = self._variables[self._factorization.variables.index(query)]

    def _initialize(self):
        # Run the algorithm until the running parameter is False
        self._running = True
        self._factor_variable_log_messages = {}
        self._variable_factor_log_messages = {}
        self._next_factors = []
        self._next_variables = []
        for factor in self.factor_leaves:
            # The leaf factor has only one variable
            variable = factor.variables[0]
            # Cache the log-message into the dictionary
            self._factor_variable_log_messages[(factor, variable)] = \
                {value: math.log(factor(value)) for value in variable.domain}
            # Add the passed factor-neighbor to the variable
            if hasattr(variable, 'passed_neighbors'):
                variable.passed_neighbors.append(factor)
            else:
                variable.passed_neighbors = [factor]
            # If all messages except one are collected,
            # then a message can be propagated from the variable
            if len(variable.passed_neighbors) + 1 == len(variable.factors):
                self._next_variables.append(variable)
        for variable in self.variable_leaves:
            if variable is self._query_variable:
                continue
            # The leaf variable has only one factor
            factor = variable.factors[0]
            # Cache the log-message into the dictionary
            self._variable_factor_log_messages[(variable, factor)] = {value: 0 for value in variable.domain}
            # Add the passed variable-neighbor to the factor
            if hasattr(factor, 'passed_neighbors'):
                factor.passed_neighbors.append(variable)
            else:
                factor.passed_neighbors = [variable]
            # If all messages except one are collected,
            # then a message can be propagated from the factor
            if len(factor.passed_neighbors) + 1 == len(factor.variables):
                self._next_factors.append(factor)