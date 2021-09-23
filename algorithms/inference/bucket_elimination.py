from pyb4ml.algorithms.inference.factored_algorithm import FactoredAlgorithm
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class BucketElimination(FactoredAlgorithm):
    def __init__(self, model: FactorGraph, elimination_order=None):
        FactoredAlgorithm.__init__(self, model)
        self._elimination_order = elimination_order
        if set(model.variables) != set(elimination_order):
            raise ValueError('the elimination order must contain the same variables as the model')

    def _resort_algorithm_variables(self):
        pass



