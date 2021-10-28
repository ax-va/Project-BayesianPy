from pyb4ml.modeling.common.named_element import NamedElement


class Factor(NamedElement):
    def __init__(self, variables, function=None, name=None):
        self._variables = tuple(variables)
        self._evidence = ()
        self._function = function
        NamedElement.__init__(self, name)
        self._link_factor_to_variables()

    def __call__(self, *variables_with_values):
        var_val_dict = dict(variables_with_values)
        values = (var_val_dict[var] for var in self._variables)
        evidential_values = {ev_var: ev_val for ev_var, ev_val in self._evidence}
        return self._function(*values, *evidential_values)

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

    def check_evidence(self, evidence):
        for ev_var, ev_val in evidence:
            self.check_variable(ev_var)
            ev_var.check_value(ev_val)

    def check_variable(self, variable):
        if variable is not self._variables:
            if isinstance(variable, Variable):
                raise ValueError(f'variable {variable.name} is not factor variable')
            else:
                raise ValueError(f'object is not a variable')

    def filter_values(self, *variables_with_values):
        return tuple(var_val for var_val in variables_with_values if var_val[0] in self._variables)

    def is_leaf(self):
        return len(self._variables) == 1

    def set_evidence(self, *evidence):
        # Check whether the evidence has duplicates
        ev_variables = tuple(ev_var for ev_var, _ in evidence)
        if len(ev_variables) != len(set(ev_variables)):
            raise ValueError(f'evidence must not contain duplicates')
        if evidence[0]:
            self.check_evidence(evidence)
            self._evidence = evidence
        else:
            self._evidence = ()

    def _link_factor_to_variables(self):
        for variable in self._variables:
            variable.link_factor(self)


if __name__ == '__main__':
    from pyb4ml.modeling.categorical.variable import Variable
    x = Variable(domain={False, True}, name='X')
    y = Variable(domain={False, True}, name='Y')
    z = Variable(domain={False, True}, name='Z')
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
