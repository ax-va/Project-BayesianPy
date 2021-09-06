from bayesian.modeling.variable import Variable


class Factor:
    def __init__(self, variables, function, link):
        self._variables = variables
        self._function = function
        self._link = link

    def __call__(self, **kwargs):
        for variable in self._variables:
            if kwargs[variable.name] not in variable.domain:
                assert False, f'Value {kwargs[variable.name]} of variable {variable} is not in its domain ' \
                              f'{variable.domain}'
        arguments = {self._link[variable_name]: kwargs[variable_name] for variable_name in kwargs.keys()}
        return self._function(**arguments)


if __name__ == '__main__':
    x = Variable(domain={False, True}, name='X')
    y = Variable(domain={False, True}, name='Y')
    z = Variable(domain={False, True}, name='Z')
    f1 = Factor(
        variables={x, y, z},
        function=lambda a, b, c: 0.5 if (a or b or c) else 0.1,
        link={x.name: 'a', y.name: 'b', z.name: 'c'}
    )
    values = {x.name: True, y.name: False, z.name: True}
    print(values)
    # print(f1(X=True, Y='777', Z=True))
    print(f1(Y=False, Z=True, X=True))
    print(f1(**values))