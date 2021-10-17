"""
The module contains the class of the Bucket Elimination Algorithm.

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

from pyb4ml.inference.factored.bucket import Bucket
from pyb4ml.inference.factored.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph
from pyb4ml.modeling.factor_graph.log_factor import LogFactor


class BE(FactoredAlgorithm):
    """
    This implementation of the Bucket Elimination (BE) algorithm works on factor graphs
    for random variables with categorical probability distributions.  That algorithm 
    belongs to Variable Elimination Algorithms.  There, a bucket contains factors used to 
    eliminate a variable by summing the product of factors over that variable.  Due to 
    that, a new factor is created and moved into one remaining bucket.  That is repeated
    until the query buckets contain factors that depend only on the query variables.  
    Instead of the factors, the implementation uses logarithms of them for computational
    stability.  See, for example, [1] for more details.  This also needs an elimination
    ordering of non-query variables.  Runtime is highly dependent on that variable
    elimination ordering, namely on the domain cardinality of free variables in a bucket
    that can be different for different orderings.  Dynamic programming here means that a
    new factor is computed only once in any run.  But in the next BE run, new factors are
    recomputed, since the possibly changed query also changes the elimination ordering.
    As a result, the ordering of computing the factors can also be changed and the factors
    computed in the previous run cannot be reused.  Moreover, although the different values 
    of evidential variables do not change the elimination ordering, they also change the 
    computed factors.  All of this makes bucket caching impractical to reuse.

    Computes a marginal (joint if necessary) probability distribution P(Q_1, ..., Q_s)
    or a conditional (joint if necessary) probability distribution
    P(Q_1, ..., Q_s | E_1 = e_1, ..., E_k = e_k), where Q_1, ..., Q_s belong to a query,
    i.e. random variables of interest, and E_1 = e_1, ..., E_k = e_k form an evidence,
    i.e. observed values e_1, ..., e_k of random variables E_1, ..., E_k, respectively.

    Restrictions:  Only works with random variables with categorical value domains.
    The factors must be strictly positive because of the use of logarithms.  The query and
    elimination variables must be disjoint.

    Recommended:  Use the algorithm for loopy factor graphs or for joint distribution of 
    query variables, otherwise use the Belief Propagation (BP) algorithm.
    
    References:

    [1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press,
    2012
    """
    def __init__(self, model: FactorGraph):
        FactoredAlgorithm.__init__(self, model)
        self._computed_log_factors = None
        self._bucket_cache = None
        self._elimination_ordering = None
        self._print_info = None
        self._name = 'Bucket Elimination'

    @property
    def ordering(self):
        return self._elimination_ordering

    def run(self, print_info=False):
        # Check whether a query is specified
        FactoredAlgorithm._is_query_set(self)
        # Check whether the query and evidence variables are disjoint
        FactoredAlgorithm._check_query_and_evidence(self)
        # Check whether an elimination order is specified
        self._is_elimination_ordering_set()
        # Check if the elimination order and query agree with each other
        self._check_query_and_elimination_variables()
        # Print the bucket information
        self._print_info = print_info
        # Clear the distribution
        self._distribution = None
        # Print info if necessary
        FactoredAlgorithm._print_start(self)
        # Initialize the bucket cache
        self._initialize_main_loop()
        # Run the main loops
        for variable in self._elimination_ordering:
            # If there are the log-factors in the output cache
            # containing that variable, they should be added into
            # the bucket of that variable
            self._add_computed_log_factors_to_bucket_cache(variable)
            # Compute the output log-factor
            # of the bucket of the variable
            # and link it in its free variables
            self._compute_output_log_factor(variable)
        for query_var in self._query:
            # If there are the log-factors in the output cache
            # containing the query variable, they should be added into
            # the bucket of the query variable
            self._add_computed_log_factors_to_bucket_cache(query_var)
        # All the output log-factors are distributed on the buckets
        # that belongs to the query variables
        self._compute_distribution()
        # Print info if necessary
        FactoredAlgorithm._print_stop(self)

    def set_ordering(self, ordering):
        # Check whether the elimination ordering has duplicates
        if len(ordering) != len(set(ordering)):
            raise ValueError(f'The elimination order must not contain duplicates')
        self._elimination_ordering = []
        # Setting the elimination ordering
        for elm_var in ordering:
            try:
                elm_var = FactoredAlgorithm._get_algorithm_variable(self, elm_var)
            except ValueError:
                self._elimination_ordering = None
                raise ValueError(f'no model variable corresponding to variable {elm_var.name!r} '
                                 f'in the elimination ordering')
            self._elimination_ordering.append(elm_var)
        self._elimination_ordering = tuple(self._elimination_ordering)

    def _add_computed_log_factors_to_bucket_cache(self, variable):
        bucket = self._bucket_cache[variable]
        remaining_log_factors = []
        for log_factor in self._computed_log_factors:
            if variable in log_factor.variables:
                bucket.add_log_factor(log_factor)
            else:
                remaining_log_factors.append(log_factor)
        self._computed_log_factors = remaining_log_factors
        # Print the bucket variable if necessary
        self._print_bucket(bucket)
        # Print the bucket input log-factors if necessary
        self._print_bucket_inputs(bucket)

    def _compute_distribution(self):
        # Assemble all the log-factors from the query buckets
        log_factors = []
        for query_variable in self._query:
            # Assemble all the log-factors from the query buckets
            log_factors.extend(self._bucket_cache[query_variable].input_log_factors)
        query_variables_values = FactoredAlgorithm.evaluate_variables(self._query)
        # Compute the function for the distribution
        nn_values = {}
        for query_values in query_variables_values:
            query_variables_with_values = tuple(zip(self._query, query_values))
            nn_values[query_values] = math.exp(
                math.fsum(
                    log_factor(
                        *log_factor.filter_values(*query_variables_with_values)) for log_factor in log_factors
                    )
                )
        # The values of the sum can be non-normalized to be the distribution.
        # The probability distribution must be normalized.
        norm_const = math.fsum(nn_values[query_values] for query_values in query_variables_values)
        # Compute the probability distribution
        self._distribution = {
            query_values:
                nn_values[query_values] / norm_const for query_values in query_variables_values
        }

    def _compute_output_log_factor(self, variable):
        # Get the variable bucket
        bucket = self._bucket_cache[variable]
        # Set free variables
        bucket.set_free_variables()
        # Compute the output log-factor of that bucket if necessary
        if bucket.has_log_factors():
            # If the bucket has no free variables, then the output log-factor is zero
            if bucket.has_free_variables():
                # Compute the output log-factor of the bucket
                log_factor = self._bucket_cache[variable].compute_output_log_factor()
                # Save the log-factor in the buffer
                self._computed_log_factors.append(log_factor)
                # Print the output log-factor if necessary
                self._print_bucket_outputs(log_factor)
        # Print the free variables if necessary
        self._print_bucket_free_variables(bucket)

    def _check_query_and_elimination_variables(self):
        set_q = set(self._query)
        set_o = set(self._elimination_ordering)
        set_m = set(self.variables)
        if not set_q.isdisjoint(set_o):
            self._query = None
            self._elimination_ordering = None
            raise ValueError('the elimination and query variables must be disjoint')
        if set_q.union(set_o) != set_m:
            self._query = None
            self._elimination_ordering = None
            raise ValueError('the elimination and query variables do not cover all the model variables')

    def _initialize_bucket_cache(self, variables):
        for variable in variables:
            self._bucket_cache[variable] = Bucket(variable)
            # Fill the variable bucket with factors
            for factor in variable.factors:
                if factor.not_added:
                    # Add the log-factor into the bucket
                    self._bucket_cache[variable].add_log_factor(LogFactor(factor))
                    # The factor is now added
                    factor.not_added = False

    def _initialize_factors(self):
        for factor in self.factors:
            factor.not_added = True

    def _initialize_main_loop(self):
        self._initialize_factors()
        self._bucket_cache = {}
        self._initialize_bucket_cache(self._elimination_ordering)
        self._initialize_bucket_cache(self._query)
        self._computed_log_factors = []

    def _is_elimination_ordering_set(self):
        # Is an elimination ordering specified?
        if self._elimination_ordering is None:
            raise AttributeError('elimination ordering not specified')

    def _print_bucket(self, bucket):
        if self._print_info:
            print()
            print(f'Bucket: {bucket.variable.name}')

    def _print_bucket_free_variables(self, bucket):
        if self._print_info:
            for free_var in bucket.free_variables:
                print('Free variable:', free_var)

    def _print_bucket_inputs(self, bucket):
        if self._print_info:
            for var in bucket.input_log_factors:
                print('Input:', var)

    def _print_bucket_outputs(self, log_factor):
        if self._print_info:
            print('Output:', log_factor)
