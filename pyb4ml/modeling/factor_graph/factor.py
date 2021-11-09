import math

from pyb4ml.modeling.categorical.variable import Variable
from pyb4ml.modeling.common.named_element import NamedElement


def log(f):
    def log_f(*t):
        return math.log(f(*t))
    return log_f


class Factor(NamedElement):
    def __init__(self, variables, function=None, name=None, evidence=None, variable_linking=True):
        NamedElement.__init__(self, name)
        self._variables = tuple(variables)
        self._function = function
        self._evidence_var_val_dict = {}
        if variable_linking:
            self._link_factor_to_variables()
        self._set_evidence(evidence)

    def __call__(self, *variables_with_values):
        var_val_dict = {}
        var_val_dict.update(self._evidence_var_val_dict)
        var_val_dict.update(dict(variables_with_values))
        values = (var_val_dict[var] for var in self._variables)
        return self._function(*values)

    def __str__(self):
        variables_names = (variable.name for variable in self._variables)
        return self._name + '(' + ', '.join(variables_names) + ')'

    @property
    def evidence(self):
        return self._evidence_var_val_dict.keys()

    @property
    def function(self):
        return self._function

    @property
    def variables(self):
        return self._variables

    @property
    def variables_number(self):
        return len(self._variables)

    def add_evidence(self, variable):
        if not isinstance(variable, Variable):
            raise ValueError(f'object {variable} is not an instance of class Variable')
        if not variable.is_evidential():
            raise ValueError(f'variable {variable.name} is not evidential')
        if variable not in self._variables:
            raise ValueError(f'variable {variable.name} does not belong to '
                             f'the factor variables {tuple(var.name for var in self._variables)}')
        self._evidence_var_val_dict[variable] = variable.domain[0]

    def clear_evidence(self):
        del self._evidence_var_val_dict
        self._evidence_var_val_dict = {}

    def delete_evidence(self, variable):
        if not isinstance(variable, Variable):
            raise ValueError(f'object {variable} is not an instance of class Variable')
        if variable not in self._variables:
            raise ValueError(f'variable {variable.name} does not belong to '
                             f'the factor variables {tuple(var.name for var in self._variables)}')
        try:
            del self._evidence_var_val_dict[variable]
        except KeyError:
            raise ValueError(f'variable {variable.name} is not evidential')

    def filter_values(self, *variables_with_values):
        return tuple(var_val for var_val in variables_with_values if var_val[0] in self._variables)

    def is_leaf(self):
        return len(self._variables) == 1

    def logarithm(self):
        self._function = log(self._function)
        self._name = 'log_' + self._name

    def _link_factor_to_variables(self):
        for var in self._variables:
            var.link_factor(self)

    def _set_evidence(self, evidence):
        self._evidence_var_val_dict = {var: var.domain[0] for var in evidence} if evidence else {}


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


