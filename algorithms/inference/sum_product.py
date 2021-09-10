import math
import itertools

from pyb4ml.algorithms.inference.algorithm import InferenceAlgorithm
from pyb4ml.algorithms.inference.messages import Message, Messages
from pyb4ml.modeling.factor_graph.factorization import Factorization
from pyb4ml.modeling.factor_graph.variable import Variable


class SumProduct(InferenceAlgorithm):
    """
    The Sum-Product Algorithm (also referred to as the Belief Propagation Algorithm)
    on factor graph trees for random variables with categorical probability
    distributions.  That belongs to Message Passing and Variable Elimination Algorithms.

    Computes a marginal probability distribution P(Q) or a conditional probability 
    distribution P(Q|E_1 = e_1, ..., E_k = e_k), where Q is a query, i.e. a random 
    variable of interest, and E_1 = e_1, ..., E_k = e_k form an evidence, i.e. observed 
    values e_1, ..., e_k of random variables E_1, ..., E_k, respectively.

    Attention: only works with categorical factors, only works on trees, leads to dead 
    lock on loopy graphs.
    
    Recommended: when modeling reduce the number of random variables in each factor to 
    speed up the inference runtime.
    """

    def __init__(self, factorization: Factorization):
        InferenceAlgorithm.__init__(self, factorization)
        # To cache the messages
        self._factor_to_variable_messages = Messages()
        self._variable_to_factor_messages = Messages()

    @staticmethod
    def _evaluate_variables(factor, fixed_variables=None, fixed_values=None):
        common_domain = []
        for index, variable in enumerate(factor.variables):
            if variable in fixed_variables:
                common_domain.append((fixed_values[index],))
            else:
                common_domain.append(variable.domain)
        return itertools.product(common_domain)

    @staticmethod
    def _extend_variable_to_factor_messages_by_zero_message(propagated_messages, non_contributed_variable):
        # Extend the propagated variable-to-factor messages by the zero message that corresponds
        # to non_contributed_variable and doesn't contribute to the sum of messages.
        # This is done in order to simplify the computation of a new message from a factor to a variable
        # that is here a non-contributed variable.
        SumProduct._zero_message.from_node = non_contributed_variable
        return tuple(propagated_messages) + (SumProduct._zero_message,)

    @staticmethod
    def _resort_variable_to_factor_messages_by_factor_variables_ordering(extended_messages, factor):
        # Resort extended variable-to-factor messages according to the variable ordering in the factor
        return (message for variable in factor.variables for message in extended_messages
                if variable is message.from_node)

    @staticmethod
    def _update_passing(from_node, to_node):
        from_node.passed = True
        to_node.incoming_messages_number += 1

    @staticmethod
    def _zero_message(value):
        return 0

    def run(self):
        # Is the query set?
        if self._query is None:
            raise AttributeError('The query was not specified')
        # Set the evidence if necessary
        # ...
        # Compute messages from leaves and make other initializations
        self._initialize_loop()
        # Running the main loop
        while True:
            # Check the stop condition
            if self._query.incoming_messages_number == self._query.factors_number:
                # Compute either the marginal or conditional probability distribution
                self._compute_distribution()
                # Break the loop
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

    def get_p(self):
        """
        Returns P(Q) or if the evidence was given then P(Q|E_1 = e_1, ..., E_k = e_k)
        as a function of q, where q is in the domain of random variable Q
        """
        if self._distribution is not None:
            def distribution(value):
                if value not in self._query.domain:
                    raise ValueError(f'The value {value!r} is not in the domain {self._query.domain}')
                return self._distribution[value]
            return distribution
        else:
            raise AttributeError('The distribution is not computed')

    def _compute_factor_to_variable_message_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            # Compute the message values
            values = {value: math.log(from_factor(value)) for value in to_variable.domain}
            # Cache the message
            self._factor_to_variable_messages.cache(Message(from_factor, to_variable, values))

    def _compute_factor_to_variable_message_not_from_leaf(self, from_factor, to_variable):
        # Compute the message if necessary
        if not self._factor_to_variable_messages.contains(from_factor, to_variable):
            # Get the incoming messages not from to_variable to from_factor
            # from_factor was previously to_factor
            variable_to_factor_messages = self._variable_to_factor_messages.get_not_from_node_to_node(
                    not_from_node=to_variable,
                    to_node=from_factor
            )
            # Used to reduce computational instability
            max_message = max(message(value) for message in variable_to_factor_messages
                              for value in message.from_node.domain)
            # Extend the propagated variable-to-factor messages by the zero message that corresponds
            # to to_variable and doesn't contribute to the sum of messages
            messages = \
                SumProduct._extend_variable_to_factor_messages_by_zero_message(variable_to_factor_messages, to_variable)
            # Resort extended variable-to-factor messages according to the variable ordering in the factor
            messages = \
                SumProduct._resort_variable_to_factor_messages_by_factor_variables_ordering(messages, from_factor)
            # Compute the message values
            values = {value:
                      max_message
                      + math.log(
                          math.fsum(
                              from_factor(eval_values)
                              * math.exp(
                                  math.fsum(
                                      msg(vls) for msg, vls in zip(messages, eval_values)
                                  ) - max_message
                              )
                              for eval_values in SumProduct._evaluate_variables(
                                  factor=from_factor,
                                  fixed_variables=(to_variable,),
                                  fixed_values=(value,))
                          )
                      ) for value in to_variable.domain}
            # Cache the message
            self._factor_to_variable_messages.cache(Message(from_factor, to_variable, values))

    def _compute_distribution(self):
        # Get the incoming messages to the query
        factor_to_query_messages = self._factor_to_variable_messages.get_from_nodes_to_node(
            from_nodes=self._query.factors,
            to_node=self._query
        )
        # Values of the sum of the incoming messages
        nn_values = {value:
            math.exp(
                math.fsum(message(value) for message in factor_to_query_messages)
            ) for value in self._query.domain}
        # The probability distribution must be normalized
        norm_const = math.fsum(nn_values[value] for value in self._query.domain)
        # Compute the probability distribution
        self._distribution = {value: nn_values[value] / norm_const for value in self._query.domain}
        self._query.passed = True

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
            # Compute the message values
            # Only one non-passed factor
            # from_variable was previously to_variable
            values = {value:
                      math.fsum(message(value) for message in
                                self._factor_to_variable_messages.get_not_from_node_to_node(
                                    not_from_node=to_factor,
                                    to_node=from_variable
                                    )
                                ) for value in from_variable.domain}
            # Cache the message
            self._variable_to_factor_messages.cache(Message(from_variable, to_factor, values))

    def _extend_next_variables(self, variable):
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
