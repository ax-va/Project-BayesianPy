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
            self._elimination_order = ...
        else:
            self._elimination_order = self._copy_elimination_order(elimination_order)
         self._factor_cache = self._factorization.create_factor_cache()

    def _copy_elimination_order(self, elimination_order):
        return tuple(self.variables[self._model.variables.index(elm_var)] for elm_var in elimination_order)

    def _remove_query_variables_from_elimination_order(self):
        self._elimination_order = list(self._elimination_order)
        for query_var in self._query:
            self._elimination_order.remove(query_var)
        self._elimination_order = tuple(self._elimination_order)











