from pyb4ml.modeling.factor_graph.node import Node
from pyb4ml.modeling.factor_graph.variable import Variable


class Factor(Node):
    def __init__(self, variables, function=None, name=None):
        self._variables = tuple(variables)
        self._function = function
        Node.__init__(self, name)
        for variable in self._variables:
            variable.link_factor(self)

    def __call__(self, *variables_with_values):
        variables_values_dict = dict(variables_with_values)
        values = (variables_values_dict[variable] for variable in self._variables)
        return self._function(*values)

    def __str__(self):
        variables_names = (variable.name for variable in self._variables)
        return self._name + '(' + ', '.join(variables_names) + ')'

    @property
    def function(self):
        return self._function

    @property
    def variables(self):
        return self._variables

    @property
    def variables_number(self):
        return len(self._variables)

    def is_leaf(self):
        return len(self._variables) == 1


if __name__ == '__main__':
    x = Variable(domain=(False, True), name='X')
    y = Variable(domain=(False, True), name='Y')
    z = Variable(domain=(False, True), name='Z')
    f1 = Factor(
        variables=(x, y, z),
        function=lambda a, b, c: 0.5 if (a or b or c) else 0.1,
        name='f1'
    )
    print(f1((x, True), (z, False), (y, True)))

    f2 = Factor(
        variables=(x, ),
        function=lambda a: 0.7 if a else 0.3,
        name='f_2'
    )
    print(f2((x, True)))
