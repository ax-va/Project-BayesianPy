from bayesian.modeling.variable import Variable


class Factor:
    def __init__(self, variables, function, link):
        self._variables = variables
        self._function = function
        self._link = link

    def __call__(self, values):
        for variable in self._variables:
            if values[variable.name] not in variable.domain:
                assert False, f'Value {values[variable.name]} of variable {variable} is not in its domain ' \
                              f'{variable.domain}'
        arguments = {self._link[variable_name]: values[variable_name] for variable_name in values.keys()}
        return self._function(**arguments)


if __name__ == '__main__':
    x = Variable(domain={False, True}) #, name='X')
    y = Variable(domain={False, True}) #, name='Y')
    z = Variable(domain={False, True}) #, name='Z')
    f1 = Factor(
        variables={x, y, z},
        function=lambda a, b, c: 0.5 if (a or b or c) else 0.1,
        link={x.name: 'a', y.name: 'b', z.name: 'c'}
    )
    values = {x.name: True, y.name: False, z.name: True}
    print(f1(values))
    print(values)
