from pyb4ml.modeling.common.named_element import NamedElement


class Variable(NamedElement):
    def __init__(self, domain=None, name=None):
        self._domain = tuple(sorted(set(domain))) if domain is not None else None
        self._linked_factors = []
        NamedElement.__init__(self, name)

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

    def check_value(self, value):
        if self.is_value_illegal(value):
            raise ValueError(f'variable {self.name} cannot have a value of {value}')

    def is_evidential(self):
        return len(self._domain) == 1

    def is_value_illegal(self, value):
        return value not in self._domain

    def is_value_legal(self, value):
        return value in self._domain

    def is_leaf(self):
        return 0 <= len(self._linked_factors) <= 1

    def link_factor(self, factor):
        self._linked_factors.append(factor)

    def set_domain(self, domain):
        self._domain = tuple(sorted(set(domain)))
