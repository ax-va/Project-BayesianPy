import itertools
import math

from pyb4ml.modeling.factor_graph.node import Node
from pyb4ml.modeling.factor_graph.variable import Variable


class Factor(Node):
    def __init__(self, variables, function, name=None):
        self._variables = variables
        self._function = function
        Node.__init__(self, name)
        for variable in self._variables:
            variable.link_factor(self)

    def __call__(self, *values):
        return self._function(*values)

    @property
    def is_leaf(self):
        return len(self._variables) == 1

    @property
    def variables(self):
        return self._variables

    @property
    def variables_number(self):
        return len(self._variables)

    def evaluate_arguments(self, fixed_variables=None, fixed_values=None):
        if fixed_variables is not None or fixed_values is not None:
            if len(fixed_variables) != len(fixed_values):
                raise ValueError("Arguments 'with_given_variables' and 'with_given_values' are not of the same size")
        common_domain = []
        for index, variable in enumerate(self._variables):
            if variable in fixed_variables:
                common_domain.append((fixed_values[index],))
            else:
                common_domain.append(variable.domain)
        return itertools.product(common_domain)


if __name__ == '__main__':
    x = Variable(domain=(False, True), name='X')
    y = Variable(domain=(False, True), name='Y')
    z = Variable(domain=(False, True), name='Z')
    f1 = Factor(
        variables=(x, y, z),
        function=lambda a, b, c: 0.5 if (a or b or c) else 0.1,
        name='f1'
    )
    print(f1(True, False, True))



