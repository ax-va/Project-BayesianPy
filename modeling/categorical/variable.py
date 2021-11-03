import itertools

from pyb4ml.modeling.common.named_element import NamedElement


class Variable(NamedElement):
    def __init__(self, domain=None, name=None):
        NamedElement.__init__(self, name)
        self._domain = tuple(sorted(set(domain))) if domain is not None else None
        self._linked_factors = []

    @staticmethod
    def evaluate_variables(variables):
        domains = (variable.domain for variable in variables)
        return tuple(itertools.product(*domains))

    @staticmethod
    def split_evidential_and_non_evidential_variables(variables, without_variables=()):
        """
        Splits evidential and non-evidential variables ignoring without_variables
        """
        evidential_variables = []
        non_evidential_variables = []
        for variable in variables:
            if variable not in without_variables:
                if variable.is_evidential():
                    evidential_variables.append(variable)
                else:
                    non_evidential_variables.append(variable)
        return tuple(evidential_variables), tuple(non_evidential_variables)

    @property
    def domain(self):
        return self._domain

    @property
    def factors(self):
        return self._linked_factors

    @property
    def factors_number(self):
        return len(self._linked_factors)

    def check_value(self, value):
        if self.is_value_illegal(value):
            raise ValueError(f'variable {self.name} cannot have the value of {value}')

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
