from pyb4ml.modeling.graph.node import Node


class Variable(Node):
    def __init__(self, domain=None, name=None):
        self._domain = tuple(sorted(set(domain))) if domain is not None else None
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

    def is_leaf(self):
        return 0 <= len(self._linked_factors) <= 1

    def link_factor(self, factor):
        self._linked_factors.append(factor)

    def set_domain(self, domain):
        self._domain = tuple(sorted(set(domain)))

    def unlink_factor(self, factor):
        if factor not in self._linked_factors:
            raise ValueError(
                f'factor {factor.name!r} not in the factors '
                f'{tuple(factor.name for factor in self._linked_factors)} linked to the variable'
            )
        self._linked_factors.remove(factor)

    def unlink_factors(self):
        self._linked_factors = []
