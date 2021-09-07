import copy
import math
from collections import Counter

from bayesian.modeling.factorization import Factorization
from bayesian.modeling.variable import Variable


class SumProduct:
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

    def run(self):
        # Startup
        # Set the evidence
        # Can you get an immediate response to the query?
        self._initialize()
        # Main loop
        while self._running:
            pass

    def set_query(self, query: Variable):
        # Single variable 'query' of interest for computing P(query) or P(query|evidence)
        self._query = self._variables[self._factorization.variables.index(query)]

    def _initialize(self):
        # Run the algorithm until the running parameter is False
        self._running = True
        # Cache the messages into the dictionary
        # The leaf factor has only one variable
        self._factor_variable_log_messages = {
            (factor, factor.variables[0]):
                {value: math.log(factor(value)) for value in factor.variables[0].domain}
            for factor in Factorization.get_factor_leaves(self._factors)
            }
        # Count the number of incoming factor-variable messages to the variable
        # The message key is a variable
        self._factor_variable_messages_number = Counter(
            map(lambda message_key: message_key[1], self._factor_variable_log_messages.keys())
        )
        # If only one factor-variable message is left, a message can be propagated from the variable
        self._next_variables = (variable for variable in self._variables
                                if len(variable.factors) - 1 == self._factor_variable_messages_number[variable])
        # Cache the messages into the dictionary
        # The leaf variable has only one factor
        self._variable_factor_log_messages = {
            (variable, variable.factors[0]):
                {value: 0 for value in variable.domain}
            for variable in Factorization.get_variable_leaves(self._variables) if variable is not self._query
        }
        # Count the number of incoming factor-variable messages to the variable
        # The message key is a factor
        self._variable_factor_messages_number = Counter(
            map(lambda message_key: message_key[1], self._variable_factor_log_messages.keys())
        )
        # If only one variable-factor message is left, a message can be propagated from the factor
        self._next_factors = (factor for factor in self._factors
                              if len(factor.variables) - 1 == self._variable_factor_messages_number[factor])

        # # To cache the log-messages into the dictionary
        # self._log_messages = {}
        # # Propagate the messages from the next factors
        # self._next_factors = []
        # # Propagate the messages from the next variables
        # self._next_variables = []
        # # Number of defined incoming factor-variable messages to the variable
        # self._factor_variable_messages_number = {variable: 0 for variable in self._variables}
        # # Number of defined incoming variable-factor messages to the factor
        # self._variable_factor_messages_number = {factor: 0 for factor in self._factors}
        # for factor in Factorization.get_factor_leaves(self._factors):
        #     # The factor-leaf has only one variable
        #     variable, = factor.variables
        #     # Cache the messages into the dictionary
        #     self._log_messages[(factor, variable)] = {value: math.log(factor(value)) for value in variable.domain}
        #     # Increment the number of incoming factor-variable messages to the variable
        #     self._factor_variable_messages_number[variable] += 1
        #     # If only one factor-variable message is left, message passing is carried out from the variable
        #     if len(variable.factors) - 1 == self._factor_variable_messages_number[variable]:
        #         # Propagate a message from this variable to the next factor
        #         self._next_variables.append(variable)
        # for variable in Factorization.get_variable_leaves(self._variables):
        #     # If the variable is the query itself, do not propagate
        #     if variable is not self._query:
        #         # The variable-leaf has only one factor
        #         factor, = variable.factors
        #         # Cache the messages into the dictionary
        #         self._log_messages[(variable, factor)] = {value: 0 for value in variable.domain}
        #         # Increment the number of incoming variable-factor messages to the factor
        #         self._variable_factor_messages_number[factor] += 1
        #         # If only one variable-factor message is left, message passing is carried out from the factor
        #         if len(factor.variables) - 1 == self._variable_factor_messages_number[factor]:
        #             # Propagate a message from this factor to the next variable
        #             self._next_factors.append(factor)