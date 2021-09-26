from pyb4ml.algorithms.inference.bucket import Bucket
from pyb4ml.algorithms.inference.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class BEA(FactoredAlgorithm):
    """
    The Bucket Elimination Algorithm
    """
    def __init__(self, model: FactorGraph, elimination_order=None):
        FactoredAlgorithm.__init__(self, model)
        if set(model.variables) != set(elimination_order):
            raise ValueError('the elimination order must contain the same variables as the model')
        if len(model.variables) != len(elimination_order):
            raise ValueError('the elimination order must contain the same number of variables as the model')
        if elimination_order is None:
            # Compute an elimination order
            self._elimination_order = ...
        else:
            self._elimination_order = self._copy_elimination_order(elimination_order)
        self._factor_cache = self._factor_graph.create_factor_cache()
        self._bucket_cache = {}

    def run(self):
        # Is a query specified?
        FactoredAlgorithm.is_query_set(self)
        # Eliminate query variables from the elimination order
        self._remove_query_variables_from_elimination_order()
        # Initialize the bucket cache
        self._initialize_main_loop()
        # Running the main loop
        for variable in self._elimination_order_without_query_variables:
            pass

    def _copy_elimination_order(self, elimination_order):
        return tuple(self.variables[self._model.variables.index(variable)] for variable in elimination_order)

    def _initialize_bucket_cache(self, variables):
        for variable in variables:
            self._bucket_cache[variable] = Bucket(variable)
            # Fill the bucket with factors and delete the factors from the factor cache
            for factor_variables in self._factor_cache.keys():
                if variable in factor_variables:
                    # Add the factor into the bucket
                    self._bucket_cache[variable].add_factor(self._factor_cache[factor_variables])
                    # Add the remaining free variables into the bucket
                    free_variables = (free_variable for free_variable in factor_variables
                                      if free_variable is not variable)
                    self._bucket_cache[variable].add_free_variables(free_variables)
                    # Reduce the factor cache by the factor
                    del self._factor_cache[factor_variables]

    def _initialize_main_loop(self):
        self._initialize_bucket_cache(self._elimination_order)
        self._initialize_bucket_cache(self._query)

    def _remove_query_variables_from_elimination_order(self):
        self._elimination_order_without_query_variables = list(self._elimination_order)
        for query_variable in self._query:
            self._elimination_order_without_query_variables.remove(query_variable)
        self._elimination_order_without_query_variables = tuple(self._elimination_order_without_query_variables)











