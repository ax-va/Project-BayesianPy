from pyb4ml.algorithms.inference.bucket import Bucket
from pyb4ml.algorithms.inference.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph
from pyb4ml.modeling.factor_graph.log_factor import LogFactor


class BEA(FactoredAlgorithm):
    """
    The Bucket Elimination Algorithm
    """
    def __init__(self, model: FactorGraph):
        FactoredAlgorithm.__init__(self, model)
        self._elimination_order = None
        self._factor_cache = self._factor_graph.create_factor_cache()
        self._bucket_cache = {}

    def is_elimination_order_set(self):
        # Is an elimination order specified?
        if self._elimination_order is None:
            raise AttributeError('elimination order not specified')

    def run(self):
        # Is a query specified?
        FactoredAlgorithm.is_query_set(self)
        # ...
        self.is_elimination_order_set()
        # ...
        self._check_elimination_order_and_query()
        # Initialize the bucket cache
        self._initialize_main_loop()
        # Run the main loop
        for variable in self._elimination_order:
            pass

    def set_elimination_order(self, elimination_order):
        # Remove duplicates if necessary
        elimination_order = set(elimination_order)
        self._elimination_order = []
        # Setting the elimination order
        for elm_var in elimination_order:
            try:
                elm_var = FactoredAlgorithm._get_algorithm_variable(self, elm_var)
            except ValueError:
                self._elimination_order = None
                raise ValueError(f'no model variable corresponding to variable {elm_var.name!r} '
                                 f'in the elimination order')
            self._elimination_order.append(elm_var)
        self._elimination_order = tuple(self._elimination_order)

    def _check_elimination_order_and_query(self):
        set_q = set(self._query)
        set_o = set(self._elimination_order)
        set_m = set(self.variables)
        if not set_q.isdisjoint(set_o):
            self._query = None
            self._elimination_order = None
            raise ValueError('the elimination and query variables must be disjoint')
        if set_q.union(set_o) != set_m:
            self._query = None
            self._elimination_order = None
            raise ValueError('the elimination and query variables do not cover all the model variables')

    def _initialize_bucket_cache(self, variables):
        for variable in variables:
            self._bucket_cache[variable] = Bucket(variable)
            # Fill the bucket with factors and delete the factors from the factor cache
            for factor_variables in self._factor_cache.keys():
                if variable in factor_variables:
                    # Add the log-factor into the bucket
                    self._bucket_cache[variable].add_log_factor(LogFactor(self._factor_cache[factor_variables]))
                    # Add the free variables of the output log-factor into the bucket
                    free_variables = (free_variable for free_variable in factor_variables
                                      if free_variable is not variable)
                    self._bucket_cache[variable].add_free_variables(free_variables)
                    # Remove the factor from the factor cache
                    del self._factor_cache[factor_variables]

    def _initialize_main_loop(self):
        self._initialize_bucket_cache(self._elimination_order)
        self._initialize_bucket_cache(self._query)











