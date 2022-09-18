from pyb4ml.inference import BE, GO
from pyb4ml.modeling import FactorGraph


class GBE(GO, BE):
    """
    Greedy Bucket Elimination (GBE)
    """
    def __init__(self, model: FactorGraph):
        GO.__init__(self, model)
        # Logarithm all the model factors
        BE._logarithm_factors(self)
        self._order_cache = {}

    def clear_order_cache(self):
        del self._order_cache
        self._order_cache = {}

    def run(self, cost='weighted-min-fill', print_info=False):
        if self._evidence in self._order_cache:
            self._elimination_order = self._order_cache[self._evidence]
        else:
            GBE._name = GO._name
            GO.run(self, cost, print_info)
            self._order_cache[self._evidence] = self._elimination_order
        GBE._name = BE._name
        BE.run(self, print_info)


if __name__ == '__main__':
    print(GBE.mro()) # GBE, GO, BE, FactoredAlgorithm, object



