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

    t = [(1, 2), (3, 4), (5, 6)]
    def f(entry):
        if entry[1] >= 4:
            return True
        return False
    print(list(filter(f, t)))

    class A:
        def __init__(self):
            self.data = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        def __contains__(self, x):
            return x == 1

        def __iter__(self):
            return iter(self.data)

        @staticmethod
        def m():
            print('hello')

    a = A()
    print(1 in a)

    print([x for x in a])
    print([x for x in a])
    I = iter(a)
    print(next(I))
    print(next(I))
    I = iter(a)
    print(next(I))
    print(next(I))
    a.m()



