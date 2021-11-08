from pyb4ml.inference import BE, GO
from pyb4ml.modeling import FactorGraph


class GBE(GO, BE):
    _name = 'Greedy Bucket Elimination'

    def __init__(self, model: FactorGraph):
        GO.__init__(self, model)
        BE._initialize_instance(self)
        self._ordering_cache = {}

    def clear_ordering_cache(self):
        del self._ordering_cache
        self._ordering_cache = {}

    def run(self, cost='weighted-min-fill', print_info=False):
        if self._evidence in self._ordering_cache:
            self._elimination_ordering = self._ordering_cache[self._evidence]
        else:
            GBE._name = GO._name
            GO.run(self, cost, print_info)
            self._ordering_cache[self._evidence] = self._elimination_ordering
        GBE._name = BE._name
        BE.run(self, print_info)


if __name__ == '__main__':
    print(GBE.mro()) # GBE, GO, BE, FactoredAlgorithm, object



