from bayesian.modeling.variable import Variable


class Factor:
    def __init__(self, variables, function):
        self._variables = variables
        self._function = function

    def __call__(self, *values):
        return self._function(*values)


if __name__ == '__main__':
    x = Variable(domain={False, True}) #, name='X')
    y = Variable(domain={False, True}) #, name='Y')
    z = Variable(domain={False, True}) #, name='Z')
    f1 = Factor(
        variables=[x, y, z],
        function=lambda a, b, c: 0.5 if (a or b or c) else 0.1,
    )
    print(f1(True, False, True))

