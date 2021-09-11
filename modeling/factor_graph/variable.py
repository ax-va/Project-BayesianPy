from pyb4ml.modeling.factor_graph.node import Node


class Variable(Node):
    def __init__(self, domain=None, name=None):
        self._domain = domain
        self._linked_factors = []
        Node.__init__(self, name)

    @property
    def domain(self):
        return self._domain

    @property
    def domain_size(self):
        return len(self._domain)

    @property
    def factors(self):
        return self._linked_factors

    @property
    def factors_number(self):
        return len(self._linked_factors)

    @property
    def name(self):
        return self._name

    def is_isolated_leaf(self):
        return len(self._linked_factors) == 0

    def is_non_isolated_leaf(self):
        return len(self._linked_factors) == 1

    def link_factor(self, factor):
        self._linked_factors.append(factor)
