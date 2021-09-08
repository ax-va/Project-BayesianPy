import math

from bayesian.algorithms.inference.factored import Factored
from bayesian.algorithms.inference.messages import Message, Messages
from bayesian.modeling.factor_graph.factorization import Factorization
from bayesian.modeling.factor_graph.variable import Variable


class SumProduct(Factored):
    """
    The Sum-Product Algorithm (also referred to as the Belief Propagation Algorithm)
    on factor graph trees for random variables with categorical probability
    distributions.  This belongs to Message Passing or Variable Elimination Algorithms.

    Computes a marginal probability distribution P(Q) or a conditional
    probability distribution P(Q|E_1 = e_1, ..., E_k = e_k), where Q is a query, i.e.
    a random variable of interest, and E_1 = e_1, ..., E_k = e_k form an evidence, i.e.
    observed values e_1, ..., e_k of random variables E_1, ..., E_k, respectively.

    Attention: only works on trees, only works with categorical factors.
    """
    def __init__(self, factorization: Factorization):
        Factored.__init__(self, factorization)
        # To cache the messages
        self._factor_to_variable_messages = Messages()
        self._variable_to_factor_messages = Messages()
        # Needed for the stop condition
        self._incoming_messages_to_query_variable = 0
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
            # Check the stop condition
            if self._incoming_messages_to_query_variable == len(self._query_variable.factors):
                # Compute either the marginal or conditional probability distribution
                ...
                # Stop the loop
                self._running = False
                break
            else:
                self._from_factors = self._next_factors
                self._next_factors = []
                self._from_variables = self._next_variables
                self._next_variables = []
                for from_factor in self._from_factors:
                    self._propagate_factor_to_variable_message_not_from_leaf(from_factor)
                for from_variable in self._from_variables:
                    self._propagate_variable_to_factor_message_not_from_leaf(from_variable)

    def set_query(self, query: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        self._query_variable = self._variables[self._factorization.variables.index(query)]

    def _compute_factor_to_variable_message_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            # Compute the message values
            values = {value: math.log(from_factor(value)) for value in to_variable.domain}
            # Cache the message
            self._factor_to_variable_messages.add(Message(from_factor, to_variable, values))

    def _compute_factor_to_variable_message_not_from_leaf(self, from_factor, to_variable):
        # from_factor was previously to_factor
        to_factor = from_factor
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            # Get the incoming messages
            variable_to_factor_messages = (message for message in self._variable_to_factor_messages
                                           if message.to_node == to_factor)
            max_message = max(message(value) for message in variable_to_factor_messages
                              for value in message.from_node.domain)
            ...

    def _compute_variable_to_factor_message_from_leaf(self, from_variable, to_factor):
        # Compute the message if necessary
        if not self._variable_to_factor_messages.contains(from_variable, to_factor):
            # Compute the message values
            values = {value: 0 for value in from_variable.domain}
            # Cache the message
            self._variable_to_factor_messages.add(Message(from_variable, to_factor, values))

    def _compute_variable_to_factor_message_not_from_leaf(self, from_variable, to_factor):
        # from_variable was previously to_variable
        to_variable = from_variable
        # Compute the message if necessary
        if not self._variable_to_factor_messages.contains(from_variable, to_factor):
            # Compute the message values
            # Only one non-passed factor
            values = {value: math.fsum(message(value) for message in self._factor_to_variable_messages
                                       if message.to_node is to_variable) for value in to_variable.domain}
            # Cache the message
            self._variable_to_factor_messages.add(Message(from_variable, to_factor, values))

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
        # Propagation from factor leaves
        self._propagate_factor_to_variable_messages_from_leaves()
        # Propagation from variable leaves
        self._propagate_variable_to_factor_messages_from_leaves()

    def _propagate_factor_to_variable_messages_from_leaves(self):
        for from_factor in self.factor_leaves:
            # The leaf factor has only one variable
            to_variable = from_factor.variables[0]
            self._compute_factor_to_variable_message_from_leaf(from_factor, to_variable)
            # If all messages except one are collected,
            # then a message can be propagated from this variable
            # to the next factor
            self._extend_next_variables(to_variable)
            if self._query_variable is to_variable:
                self._incoming_messages_to_query_variable += 1

    def _propagate_factor_to_variable_message_not_from_leaf(self, from_factor):
        # The factor-to-variable message to the variable
        to_variable, = (variable for variable in from_factor.variables
                        if variable not in from_factor.passed_neighbors)
        self._compute_factor_to_variable_message_not_from_leaf(from_factor, to_variable)
        # Append the passed factor-neighbor to the variable
        to_variable.passed_neighbors.append(from_factor)
        # If all messages except one are collected,
        # then a message can be propagated from the next factor
        # to the next variable
        self._extend_next_variables(to_variable)
        if self._query_variable is to_variable:
            self._incoming_messages_to_query_variable += 1

    def _propagate_variable_to_factor_messages_from_leaves(self):
        for from_variable in self.variable_leaves:
            if from_variable is self._query_variable:
                continue
            # The leaf variable has only one factor
            to_factor = from_variable.factors[0]
            self._compute_variable_to_factor_message_from_leaf(from_variable, to_factor)
            # If all messages except one are collected,
            # then a message can be propagated from this factor
            # to the next variable
            self._extend_next_factors(to_factor)

    def _propagate_variable_to_factor_message_not_from_leaf(self, from_variable):
        # The variable-to-factor message to the factor
        to_factor, = (factor for factor in from_variable.factors
                      if factor not in from_variable.passed_neighbors)
        self._compute_variable_to_factor_message_not_from_leaf(from_variable, to_factor)
        # Append the passed variable-neighbor to the factor
        to_factor.passed_neighbors.append(from_variable)
        # If all messages except one are collected,
        # then a message can be propagated from the next factor
        # to the next variable
        self._extend_next_factors(to_factor)