import pyb4ml.modeling.utils.checks as checks
import pyb4ml.modeling.utils.logarithm as logarithm
from pyb4ml.modeling.elements.model_element import ModelElement


class Factor(ModelElement):
    def __init__(self, variables, name, function=None, evidential=None):
        ModelElement.__init__(self, name)
        self._variables = tuple(variables)
        self._update_variable_factors()
        self._function = function
        self._evidential = tuple(evidential) if evidential else ()
        self._evidence_dict = {var: var.domain[0] for var in self._evidential}
        try:
            checks.check_are_variables_evidential(self._evidential)
            checks.check_disjoint(self._variables, self._evidential)
        except Exception as e:
            self.clear_evidential()
            raise e

    def __call__(self, *var_val_args):
        var_val_dict = {}
        var_val_dict.update(self._evidence_dict)
        var_val_dict.update(dict(var_val_args))
        values = (var_val_dict[var] for var in self._variables)
        return self._function(*values)

    def __str__(self):
        var_names = (var.name for var in self._variables)
        return self._name + '(' + ', '.join(var_names) + ')'

    @property
    def evidential(self):
        return self._evidential

    @property
    def evidence(self):
        return tuple(self._evidence_dict)

    @property
    def function(self):
        return self._function

    @property
    def variables(self):
        return self._variables

    @property
    def var_number(self):
        return len(self._variables)

    def check_in_variables(self, variable):
        if variable not in self._variables:
            raise ValueError(f"Variable '{variable.name}' is not in "
                             f"the free factor variables {tuple(var.name for var in self._variables)}")

    def check_in_evidential(self, variable):
        if variable not in self._evidential:
            raise ValueError(f"Variable '{variable.name}' is not in "
                             f"the evidential factor variables {tuple(var.name for var in self._variables)}")

    def add_to_evidential(self, variable):
        checks.check_variable_instance(variable)
        checks.check_is_variable_evidential(variable)
        self.check_in_variables(variable)
        self._variables = list(self._variables)
        self._variables.remove(variable)
        self._variables = tuple(self._variables)
        self._evidential = list(self._evidential)
        self._evidential.append(variable)
        self._evidential = tuple(self._evidential)
        self._evidence_dict[variable] = variable.domain[0]

    def clear_evidential(self):
        del self._evidential
        del self._evidence_dict
        self._evidential = ()
        self._evidence_dict = {}

    def delete_from_evidential(self, variable):
        checks.check_variable_instance(variable)
        checks.check_is_variable_non_evidential(variable)
        self.check_in_evidential(variable)
        self._variables = list(self._variables)
        self._variables.append(variable)
        self._variables = tuple(self._variables)
        self._evidential = list(self._evidential)
        self._evidential.remove(variable)
        self._evidential = tuple(self._evidential)
        del self._evidence_dict[variable]

    def filter_call_arguments(self, *var_val_args):
        return tuple(var_val for var_val in var_val_args if var_val[0] in self._variables)

    def is_leaf(self):
        return len(self._variables) + len(self._evidential) == 1

    def logarithm(self):
        self._function = logarithm.logarithm(self._function)
        self._name = 'log_' + self._name

    def _update_variable_factors(self):
        for var in self._variables:
            var.add_factor(self)


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


