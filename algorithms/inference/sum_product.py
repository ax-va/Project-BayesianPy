import math
import itertools

from pyb4ml.algorithms.inference.algorithm import InferenceAlgorithm
from pyb4ml.algorithms.inference.node_to_node_messages import Message, Messages
from pyb4ml.models.factor_graphs.model import Model


class SumProduct(InferenceAlgorithm):
    """
    The Sum-Product Algorithm (also referred to as the Belief Propagation Algorithm)
    works on factor-graph trees for random variables with categorical probability
    distributions.  That belongs to Message Passing and Variable Elimination Algorithms.
    Here, the factor-to-variable and variable-to-factor messages are propagated from the
    leaves across the tree to the query variable.  That can be considered as the successive 
    elimination of the factors and variables in the factor graph.  Instead of the messages, 
    the implementation uses logarithms of them for computational stability.  See, for
    example, [1] for more details.

    Computes a marginal probability distribution P(Q) or a conditional probability 
    distribution P(Q|E_1 = e_1, ..., E_k = e_k), where Q is a query, i.e. a random 
    variable of interest, and E_1 = e_1, ..., E_k = e_k form an evidence, i.e. observed 
    values e_1, ..., e_k of random variables E_1, ..., E_k, respectively.

    Restrictions: only works with categorical random variables, only works on trees, leads 
    to dead lock on loopy graphs.  The factors must be strictly positive because of the use 
    of logarithms.
    
    Recommended: when modeling, reduce the number of random variables in each factor to 
    speed up the inference runtime.

    [1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press,
    2012
    """

    def __init__(self, model: Model):
        InferenceAlgorithm.__init__(self, model)
        # To cache the node-to-node messages
        self._factor_to_variable_messages = Messages()
        self._variable_to_factor_messages = Messages()
        # Whether to print propagating node-to-node messages
        self._print_messages = False
        # Whether to print loop passing
        self._print_loop_passing = False
        self._from_factors = None
        self._next_factors = None
        self._from_variables = None
        self._next_variables = None

    @staticmethod
    def _evaluate_variables(factor, fixed_variables=None, fixed_values=None):
        common_domain = []
        index = 0
        for variable in factor.variables:
            if variable in fixed_variables:
                common_domain.append((fixed_values[index], ))
                index += 1
            else:
                common_domain.append(variable.domain)
        return itertools.product(*common_domain)

    @staticmethod
    def _extend_messages_by_zero_message(propagated_messages, non_contributed_variable):
        # Extend the propagated variable-to-factor messages by the zero message that corresponds
        # to non_contributed_variable and doesn't contribute to the sum of messages.
        # This is done in order to simplify the computation of a new message from a factor to a variable
        # that is here a non-contributed variable.
        SumProduct._zero_message.from_node = non_contributed_variable
        return tuple(propagated_messages) + (SumProduct._zero_message, )

    @staticmethod
    def _resort_messages_by_factor_variables_ordering(extended_messages, factor):
        # Resort extended variable-to-factor messages according to the variable ordering in the factor
        return tuple(message for variable in factor.variables for message in extended_messages
                     if variable is message.from_node)

    @staticmethod
    def _update_passing(from_node, to_node):
        from_node.passed = True
        to_node.incoming_messages_number += 1

    @staticmethod
    def _zero_message(value):
        return 0

    def run(self, print_messages=False, print_loop_passing=False):
        # Whether to print propagating messages
        self._print_messages = print_messages
        # Whether to print loop passing
        self._print_loop_passing = print_loop_passing
        self._distribution = None
        # Is the query set?
        if self._query is None:
            raise AttributeError('query not specified')
        # Evidence?
        # ...
        # Compute messages from leaves and make other initializations
        self._initialize_loop()
        # Running the main loop
        while True:
            self._loop_passing += 1
            # Print the number of the main-loop passes
            self._print_loop()
            # Check the stop condition
            if self._query.incoming_messages_number == self._query.factors_number:
                # Compute either the marginal or conditional probability distribution
                self._compute_distribution()
                self._print_stop()
                # Break the main loop
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

    def _compute_distribution(self):
        # Get the incoming messages to the query
        factor_to_query_messages = self._factor_to_variable_messages.get_from_nodes_to_node(
            from_nodes=self._query.factors,
            to_node=self._query
        )
        # Values of the sum of the incoming messages,
        # yet non-normalized to be the disribution
        nn_values = {value:
                     math.exp(
                         math.fsum(message(value) for message in factor_to_query_messages)
            ) for value in self._query.domain}
        # The probability distribution must be normalized
        norm_const = math.fsum(nn_values[value] for value in self._query.domain)
        # Compute the probability distribution
        self._distribution = {value: nn_values[value] / norm_const for value in self._query.domain}
        self._query.passed = True

    def _compute_factor_to_variable_message_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            # Compute the message values
            for value in to_variable.domain:
                values = {value: math.log(from_factor((value, ))) for value in to_variable.domain}
            # Cache the message
            self._factor_to_variable_messages.cache(Message(from_factor, to_variable, values))

    def _compute_factor_to_variable_message_not_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            from_variables = tuple(variable for variable in from_factor.variables if variable is not to_variable)
            # Get the incoming messages not from to_variable to from_factor
            # from_factor was previously to_factor
            messages0 = self._variable_to_factor_messages.get_from_nodes_to_node(
                from_nodes=from_variables,
                to_node=from_factor
            )
            # Used to reduce computational instability
            max_message = max(message(value) for message in messages0 for value in message.from_node.domain)
            # Extend the propagated variable-to-factor messages by the zero message that corresponds
            # to to_variable and doesn't contribute to the sum of messages
            messages1 = SumProduct._extend_messages_by_zero_message(messages0, to_variable)
            # Resort extended variable-to-factor messages according to the variable ordering in the factor
            messages2 = SumProduct._resort_messages_by_factor_variables_ordering(messages1, from_factor)
            # Compute the message values
            values = {value:
                      max_message
                      + math.log(
                          math.fsum(
                              from_factor(eval_values)
                              * math.exp(
                                  math.fsum(
                                      msg(vls) for msg, vls in zip(messages2, eval_values)
                                  ) - max_message
                              )
                              for eval_values in SumProduct._evaluate_variables(
                                  factor=from_factor,
                                  fixed_variables=(to_variable, ),
                                  fixed_values=(value, )
                              )
                          )
                      ) for value in to_variable.domain}
            # Cache the message
            self._factor_to_variable_messages.cache(Message(from_factor, to_variable, values))

    def _compute_variable_to_factor_message_from_leaf(self, from_variable, to_factor):
        # Compute the message if necessary
        if not self._variable_to_factor_messages.contains(from_variable, to_factor):
            # Compute the message values
            values = {value: 0 for value in from_variable.domain}
            # Cache the message
            self._variable_to_factor_messages.cache(Message(from_variable, to_factor, values))

    def _compute_variable_to_factor_message_not_from_leaf(self, from_variable, to_factor):
        # Compute the message if necessary
        if not self._variable_to_factor_messages.contains(from_variable, to_factor):
            from_factors = tuple(factor for factor in from_variable.factors if factor is not to_factor)
            # Compute the message values
            # Only one non-passed factor
            # from_variable was previously to_variable
            values = {value:
                      math.fsum(message(value) for message in
                                self._factor_to_variable_messages.get_from_nodes_to_node(
                                    from_nodes=from_factors,
                                    to_node=from_variable)
                                ) for value in from_variable.domain}
            # Cache the message
            self._variable_to_factor_messages.cache(Message(from_variable, to_factor, values))

    def _extend_next_variables(self, variable):
        # If the variable is query, the propagation should be stopped here
        if variable is not self._query:
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

    def _initialize_factor_passing(self):
        # There are no passed factors
        for factor in self._factors:
            factor.passed = False
            factor.incoming_messages_number = 0

    def _initialize_loop(self):
        self._loop_passing = 0
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
        for variable in self._variables:
            variable.passed = False
            variable.incoming_messages_number = 0

    def _propagate_factor_to_variable_messages_from_leaves(self):
        for from_factor in self.factor_leaves:
            # The leaf factor has only one variable
            to_variable = from_factor.variables[0]
            self._compute_factor_to_variable_message_from_leaf(from_factor, to_variable)
            # Update passed nodes und incoming messages number
            self._update_passing(from_factor, to_variable)
            # If all messages except one are collected,
            # then a message can be propagated from this variable
            # to the next factor
            self._extend_next_variables(to_variable)
            # Print the message if necessary
            self._print_message(from_factor, to_variable, self._factor_to_variable_messages)

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
        # Print the message if necessary
        self._print_message(from_factor, to_variable, self._factor_to_variable_messages)

    def _propagate_variable_to_factor_messages_from_leaves(self):
        for from_variable in self.variable_leaves:
            if from_variable is self._query:
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
            # Print the message if necessary
            self._print_message(from_variable, to_factor, self._variable_to_factor_messages)

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
        # Print the message if necessary
        self._print_message(from_variable, to_factor, self._variable_to_factor_messages)

    def _print_loop(self):
        if self._print_loop_passing:
            print('loop passing:', self._loop_passing)
            print()

    def _print_message(self, from_node, to_node, messages):
        # Print the message if necessary
        if self._print_messages:
            print(messages.get(from_node, to_node))
            print('logarithmic message value:')
            print(messages.get(from_node, to_node).values)
            print('message values:')
            print({key: math.exp(value) for key, value in messages.get(from_node, to_node).values.items()})
            print()
            
    def _print_stop(self):
        if self._print_loop_passing:
            print('algorithm stopped')
