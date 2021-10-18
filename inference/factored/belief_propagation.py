"""
The module contains the class of the Belief Propagation Algorithm.

Attention:  The author is not responsible for any damage that can be caused by the use
of this code.  You use this code at your own risk.  Any claim against the author is 
legally void.  By using this code, you agree to the terms imposed by the author.

Achtung:  Der Autor haftet nicht für Schäden, die durch die Verwendung dieses Codes
entstehen können.  Sie verwenden dieses Code auf eigene Gefahr.  Jegliche Ansprüche 
gegen den Autor sind rechtlich nichtig.  Durch die Verwendung dieses Codes stimmen 
Sie den vom Autor auferlegten Bedingungen zu.

© 2021 Alexander Vasiliev
"""
import math

from pyb4ml.inference.factored.factored_algorithm import FactoredAlgorithm
from pyb4ml.inference.factored.factor_tree_messages import Message, Messages
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class BP(FactoredAlgorithm):
    """
    This implementation of the Belief Propagation (BP) algorithm works on factor graph
    trees for random variables with categorical probability distributions.  That algorithm 
    belongs to Message Passing and Variable Elimination Algorithms.  There, the 
    factor-to-variable and variable-to-factor messages are propagated from the leaves across
    the tree to the query variable.  That can be considered as the successive elimination
    of the factors and variables in the factor graph tree.  This implementation encourages
    reuse of the algorithm by caching already computed messages given an evidence or no 
    evidence.  Thus, they are computed only once, which is dynamic programming, and are used
    in the next BPA runs.  Instead of the messages, the implementation uses logarithms of 
    them for computational stability.  See, for example, [1] for more details.
    
    Computes a marginal probability distribution P(Q) or a conditional probability 
    distribution P(Q | E_1 = e_1, ..., E_k = e_k), where Q is a query, i.e. a random
    variable of interest, and E_1 = e_1, ..., E_k = e_k form an evidence, i.e. observed 
    values e_1, ..., e_k of random variables E_1, ..., E_k, respectively.

    Restrictions:  Only works with random variables with categorical value domains, only 
    works on trees (leads to dead lock on loopy graphs).  See the Bucket Elimination (BE)
    algorithm for the case of loopy graphs or joint distribution of query variables.
    The factors must be strictly positive because of the use of logarithms.
    
    Recommended:  When modeling, reduce the number of random variables in each factor to 
    speed up the inference runtime.  To reduce the number of variables in factors, you can, 
    for example, increase the number of variables and factors in a model.

    References:

    [1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press,
    2012
    """

    def __init__(self, model: FactorGraph):
        FactoredAlgorithm.__init__(self, model)
        # To cache the node-to-node messages
        self._factor_to_variable_messages = {}
        self._variable_to_factor_messages = {}
        # Query variable
        self._query_variable = None
        # Whether to print loop passing and propagating node-to-node messages
        self._print_info = None
        # Temporary buffers
        self._from_factors = None
        self._next_factors = None
        self._from_variables = None
        self._next_variables = None
        self._name = 'Belief Propagation'

    @staticmethod
    def _update_passing(from_node, to_node):
        from_node.passed = True
        to_node.incoming_messages_number += 1

    def clear_cached_messages(self):
        del self._factor_to_variable_messages
        del self._variable_to_factor_messages
        self._factor_to_variable_messages = {}
        self._variable_to_factor_messages = {}

    def run(self, print_info=False):
        # Check whether a query is specified
        FactoredAlgorithm._is_query_set(self)
        # Check whether the query and evidence variables are disjoint
        FactoredAlgorithm._check_query_and_evidence(self)
        # Check whether the query has only one variable
        FactoredAlgorithm._has_query_only_one_variable(self)
        # Set the first variable to the query
        self._query_variable = self._query[0]
        # The message caching is based on evidence
        self._create_factor_to_variable_messages_cache_if_necessary()
        # The message caching is based on evidence
        self._create_variable_to_factor_messages_cache_if_necessary()
        # Whether to print loop passing and propagating node-to-node messages
        self._print_info = print_info
        # Clear the distribution
        self._distribution = None
        # Print info if necessary
        FactoredAlgorithm._print_start(self)
        # Compute messages from leaves and make other initializations
        self._initialize_main_loop()
        # Run the main loop
        # Stop condition: self._query_variable.incoming_messages_number == self._query_variable.factors_number
        while self._get_running_condition():
            self._loop_passing += 1
            # Print the number of the main-loop passes
            self._print_loop()
            # Next initialization
            self._from_factors = self._next_factors
            self._next_factors = []
            self._from_variables = self._next_variables
            self._next_variables = []
            for from_factor in self._from_factors:
                self._propagate_factor_to_variable_message_not_from_leaf(from_factor)
            for from_variable in self._from_variables:
                self._propagate_variable_to_factor_message_not_from_leaf(from_variable)
        # Propagation stopped
        # Compute either the marginal or conditional probability distribution
        self._compute_distribution()
        # Print info if necessary
        FactoredAlgorithm._print_stop(self)

    def _compute_distribution(self):
        # Get the incoming messages to the query
        factor_to_query_messages = self._factor_to_variable_messages[self._evidence].get_from_nodes_to_node(
            from_nodes=self._query_variable.factors,
            to_node=self._query_variable
        )
        # Compute the function for the distribution
        nn_values = {value: math.exp(
            math.fsum(message(value) for message in factor_to_query_messages)
        ) for value in self._query_variable.domain}
        # The values of the sum of the incoming messages
        # can be non-normalized to be the distribution.
        # The probability distribution must be normalized.
        norm_const = math.fsum(nn_values[value] for value in self._query_variable.domain)
        # Compute the probability distribution
        self._distribution = {(value, ): nn_values[value] / norm_const for value in self._query_variable.domain}
        self._query_variable.passed = True

    def _compute_factor_to_variable_message_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages[self._evidence].contains(from_factor, to_variable):
            # Compute the message values
            values = {value: math.log(from_factor((to_variable, value))) for value in to_variable.domain}
            # Cache the message
            message = Message(from_factor, to_variable, values)
            self._factor_to_variable_messages[self._evidence].cache(message)
            # Print the message if necessary
            self._print_message(message)

    def _compute_factor_to_variable_message_not_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages[self._evidence].contains(from_factor, to_variable):
            # Split evidential and non-evidential variables
            from_evidential_variables, from_non_evidential_variables = \
                FactoredAlgorithm.split_evidential_and_non_evidential_variables(
                    variables=from_factor.variables,
                    without_variables=(to_variable, )
                )
            # Get the incoming evidential messages
            evidential_messages = self._variable_to_factor_messages[self._evidence].get_from_nodes_to_node(
                from_nodes=from_evidential_variables,
                to_node=from_factor
            )
            # Get the incoming non-evidential messages
            non_evidential_messages = self._variable_to_factor_messages[self._evidence].get_from_nodes_to_node(
                from_nodes=from_non_evidential_variables,
                to_node=from_factor
            )
            # Use to reduce computational instability
            max_message = \
                max(message(value) for message in non_evidential_messages for value in message.from_node.domain) \
                if len(non_evidential_messages) > 0 else 0
            # Sum out the evidential messages separately
            from_evidential_variables_values = tuple(variable.domain[0] for variable in from_evidential_variables)
            evidential_variables_with_values = tuple(zip(from_evidential_variables, from_evidential_variables_values))
            evidential_messages_sum = math.fsum(msg(val) for msg, val
                                                in zip(evidential_messages, from_evidential_variables_values))
            # Cross product of domains
            evaluated_non_evidential_variables_values = \
                FactoredAlgorithm.evaluate_variables(from_non_evidential_variables)
            # Compute the message values
            values = {value: evidential_messages_sum + max_message + math.log(
                              math.fsum(
                                  from_factor(
                                      *zip(from_non_evidential_variables, non_evidential_variables_values),
                                      *evidential_variables_with_values,
                                      (to_variable, value)
                                  )
                                  * math.exp(
                                      math.fsum(
                                          msg(val) for msg, val
                                          in zip(non_evidential_messages, non_evidential_variables_values)
                                      ) - max_message
                                  )
                                  for non_evidential_variables_values in evaluated_non_evidential_variables_values
                              )
                          ) for value in to_variable.domain}
            # Cache the message
            message = Message(from_factor, to_variable, values)
            self._factor_to_variable_messages[self._evidence].cache(message)
            # Print the message if necessary
            self._print_message(message)

    def _compute_variable_to_factor_message_from_leaf(self, from_variable, to_factor):
        # Compute the message if necessary
        if not self._variable_to_factor_messages[self._evidence].contains(from_variable, to_factor):
            # Compute the message values
            values = {value: 0 for value in from_variable.domain}
            # Cache the message
            message = Message(from_variable, to_factor, values)
            self._variable_to_factor_messages[self._evidence].cache(message)
            # Print the message if necessary
            self._print_message(message)

    def _compute_variable_to_factor_message_not_from_leaf(self, from_variable, to_factor):
        # Compute the message if necessary
        if not self._variable_to_factor_messages[self._evidence].contains(from_variable, to_factor):
            from_factors = tuple(factor for factor in from_variable.factors if factor is not to_factor)
            # Compute the message values
            # Only one non-passed factor
            # from_variable was previously to_variable
            values = {value: math.fsum(message(value) for message in
                                       self._factor_to_variable_messages[self._evidence].get_from_nodes_to_node(
                                           from_nodes=from_factors,
                                           to_node=from_variable)
                                       ) for value in from_variable.domain}
            # Cache the message
            message = Message(from_variable, to_factor, values)
            self._variable_to_factor_messages[self._evidence].cache(message)
            # Print the message if necessary
            self._print_message(message)

    def _create_factor_to_variable_messages_cache_if_necessary(self):
        if self._evidence not in self._factor_to_variable_messages:
            # Cache if not cached
            self._factor_to_variable_messages[self._evidence] = Messages()

    def _create_variable_to_factor_messages_cache_if_necessary(self):
        if self._evidence not in self._variable_to_factor_messages:
            # Cache if not cached
            self._variable_to_factor_messages[self._evidence] = Messages()

    def _extend_next_variables(self, variable):
        # If the variable is query, the propagation should be stopped here
        if variable is not self._query_variable:
            # If all messages except one are collected,
            # then a message can be propagated from this variable
            # to the next factor
            if variable.incoming_messages_number + 1 == variable.factors_number:
                self._next_variables.append(variable)

    def _extend_next_factors(self, factor):
        # If all messages except one are collected,
        # then a message can be propagated from this factor
        # to the next variable
        if factor.incoming_messages_number + 1 == factor.variables_number:
            self._next_factors.append(factor)

    def _get_running_condition(self):
        return self._query_variable.incoming_messages_number < self._query_variable.factors_number

    def _initialize_factor_passing(self):
        # There are no passed factors
        for factor in self.factors:
            factor.passed = False
            factor.incoming_messages_number = 0

    def _initialize_main_loop(self):
        self._loop_passing = 0
        # Print the loop information if necessary
        self._print_loop()
        # The factors to which the message propagation goes further
        self._next_factors = []
        # The variables to which the message propagation goes further
        self._next_variables = []
        # There are no passed factors and no incoming messages
        self._initialize_factor_passing()
        # There are no passed variables and no incoming messages
        self._initialize_variable_passing()
        # Propagation from factor leaves
        self._propagate_factor_to_variable_messages_from_leaves()
        # Propagation from variable leaves
        self._propagate_variable_to_factor_messages_from_leaves()

    def _initialize_variable_passing(self):
        # There are no passed variables
        for variable in self.variables:
            variable.passed = False
            variable.incoming_messages_number = 0

    def _propagate_factor_to_variable_messages_from_leaves(self):
        for from_factor in self._factor_graph.factor_leaves:
            # The leaf factor has only one variable
            to_variable = from_factor.variables[0]
            self._compute_factor_to_variable_message_from_leaf(from_factor, to_variable)
            # Update passed nodes und incoming messages number
            self._update_passing(from_factor, to_variable)
            # If all messages except one are collected,
            # then a message can be propagated from this variable
            # to the next factor
            self._extend_next_variables(to_variable)

    def _propagate_factor_to_variable_message_not_from_leaf(self, from_factor):
        # The factor-to-variable message to the only one variable that is non-passed
        to_variable, = (variable for variable in from_factor.variables if not variable.passed)
        self._compute_factor_to_variable_message_not_from_leaf(from_factor, to_variable)
        # Update passed nodes und incoming messages number
        self._update_passing(from_factor, to_variable)
        # If all messages except one are collected,
        # then a message can be propagated from the next factor
        # to the next variable
        self._extend_next_variables(to_variable)

    def _propagate_variable_to_factor_messages_from_leaves(self):
        for from_variable in self._factor_graph.variable_leaves:
            if from_variable is self._query_variable:
                continue
            # The leaf variable has only one factor
            to_factor = from_variable.factors[0]
            self._compute_variable_to_factor_message_from_leaf(from_variable, to_factor)
            # Update passed nodes und incoming messages number
            self._update_passing(from_variable, to_factor)
            # If all messages except one are collected,
            # then a message can be propagated from this factor
            # to the next variable
            self._extend_next_factors(to_factor)

    def _propagate_variable_to_factor_message_not_from_leaf(self, from_variable):
        # The variable-to-factor message to the only one factor that is non-passed
        to_factor, = (factor for factor in from_variable.factors if not factor.passed)
        self._compute_variable_to_factor_message_not_from_leaf(from_variable, to_factor)
        # Update passed nodes und incoming messages number
        self._update_passing(from_variable, to_factor)
        # If all messages except one are collected,
        # then a message can be propagated from the next factor
        # to the next variable
        self._extend_next_factors(to_factor)

    def _print_loop(self):
        if self._print_info:
            print()
            print('loop passing:', self._loop_passing)
            print()

    def _print_message(self, message):
        # Print the message if necessary
        if self._print_info:
            print(message)
            print('logarithmic message value:')
            print(message.values)
            print('message values:')
            print({key: math.exp(value) for key, value in message.values.items()})
