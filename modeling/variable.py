class Variable:
    def __init__(self, domain):
        self._domain = domain
        self._linked_factors = []

    @property
    def domain(self):
        return self._domain

    def link_factor(self, factor):
        self._linked_factors.append(factor)
