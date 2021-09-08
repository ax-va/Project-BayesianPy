import math

from bayesian.algorithms.inference.factored import Factored
from bayesian.modeling.factor_graph.factorization import Factorization
from bayesian.modeling.factor_graph.variable import Variable


class SumProduct(Factored):
    """
    The Sum-Product Algorithm (also referred to as the Belief Propagation Algorithm)
    on factor graph trees for random variables with categorical probability
    distributions.  This belongs to Message Passing or Variable Elimination Algorithms.

    Computes a marginal probability distribution P(Q) or a conditional
    probability distribution P(Q|E_1 = e_1, ..., E_k = e_k), where Q is a query, i.e.
    random variable of interest, and E_1 = e_1, ..., E_k = e_k form an evidence, i.e.
    observed values of random variables.
    """
    def __init__(self, factorization: Factorization):
        Factored.__init__(self, factorization)
        # Cache the log-messages into the dictionaries
        self._factor_to_variable_messages = {}
        self._variable_to_factor_messages = {}
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
            # Stop condition
            if len(self._query_variable.passed_neighbors) == len(self._query_variable.factors):
                # Compute either the marginal or conditional probability distribution
                # ...
                # Stop the loop
                self._running = False
            else:
                self._from_factors = self._next_factors
                self._next_factors = []
                self._from_variables = self._next_variables
                self._next_variables = []
                for from_factor in self._from_factors:
                    pass

                for from_variable in self._from_variables:
                    # Previous to_variable is current from_variable
                    to_variable = from_variable
                    # The log-message from this variable to the only one non-passed factor
                    variable_to_factor_message = {
                        value: math.fsum(
                            message[value] for message in (
                                self._factor_to_variable_messages[(from_factor, to_variable)]
                                for from_factor in to_variable.passed_neighbors
                            )
                        )
                        for value in to_variable.domain
                    }
                    # The variable-to-factor message to the factor
                    to_factor, = (factor for factor in from_variable.factors
                                  if factor not in from_variable.passed_neighbors)
                    # Cache the log-message into the dictionary
                    self._variable_to_factor_messages[(from_variable, to_factor)] = variable_to_factor_message
                    # Append the passed variable-neighbor to the next factor
                    to_factor.passed_neighbors.append(from_variable)
                    # If all messages except one are collected,
                    # then a message can be propagated from the next factor
                    # to the next variable
                    self._extend_next_factors(to_factor)

    def set_query(self, query: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        self._query_variable = self._variables[self._factorization.variables.index(query)]

    def _extend_next_variables(self, variable):
        # If all messages except one are collected,
        # then a message can be propagated from this variable
        # to the next factor
        if len(variable.passed_neighbors) + 1 == len(variable.factors):
            self._next_variables.append(variable)

    def _extend_next_factors(self, factor):
        # If all messages except one are collected,
        # then a message can be propagated from this factor
        # to the next variable
        if len(factor.passed_neighbors) + 1 == len(factor.variables):
            self._next_factors.append(factor)

    def _initialize(self):
        # Run the algorithm until the running parameter is False
        self._running = True
        # The factors to which the message propagation goes further
        self._next_factors = []
        # The variables to which the message propagation goes further
        self._next_variables = []
        # Append leer passed neighbors to the factors and variables
        self._initialize_passed_neighbors()
        # Propagation from factor leaves
        self._initialize_over_factor_leaves()
        # Propagation from variable leaves
        self._initialize_over_variable_leaves()

    def _initialize_over_factor_leaves(self):
        for from_factor in self.factor_leaves:
            # The leaf factor has only one variable
            to_variable = from_factor.variables[0]
            # Compute the message
            factor_to_variable_message = {value: math.log(from_factor(value)) for value in to_variable.domain}
            # Cache the log-message into the dictionary
            self._factor_to_variable_messages[(from_factor, to_variable)] = factor_to_variable_message
            # Append the passed factor-neighbor to the variable
            to_variable.passed_neighbors.append(from_factor)
            # If all messages except one are collected,
            # then a message can be propagated from this variable
            # to the next factor
            self._extend_next_variables(to_variable)

    def _initialize_over_variable_leaves(self):
        for from_variable in self.variable_leaves:
            if from_variable is self._query_variable:
                continue
            # The leaf variable has only one factor
            to_factor = from_variable.factors[0]
            # Compute the message
            variable_to_factor_message = {value: 0 for value in from_variable.domain}
            # Cache the log-message into the dictionary
            self._variable_to_factor_messages[(from_variable, to_factor)] = variable_to_factor_message
            # Append the passed variable-neighbor to the factor
            to_factor.passed_neighbors.append(from_variable)
            # If all messages except one are collected,
            # then a message can be propagated from this factor
            # to the next variable
            self._extend_next_factors(to_factor)

    def _initialize_passed_neighbors(self):
        # Append leer passed neighbors to the factors and variables
        for factor in self._factors:
            factor.passed_neighbors = []
        for variable in self._variables:
            variable.passed_neighbors = []