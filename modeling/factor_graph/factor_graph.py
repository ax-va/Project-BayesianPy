class FactorGraph:
    def __init__(self, factors):
        self._factors = sorted(
            set(factors),
            key=lambda f: f.name
        )
        self._variables = sorted(
            set(variable
                for factor in self._factors
                for variable in factor.variables
                ),
            key=lambda v: v.name
        )
        self._factor_dict = None
        self._variable_dict = None

    @property
    def factors(self):
        return self._factors

    @property
    def factor_leaves(self):
        return tuple(factor for factor in self._factors if factor.is_leaf())

    @property
    def factor_variables(self):
        return tuple(factor.variables for factor in self._factors)

    @property
    def variables(self):
        return self._variables

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_leaf())

    def get_factor(self, name):
        if self._factor_dict is None:
            self._set_factor_dict()
        try:
            return self._factor_dict[name]
        except KeyError:
            raise AttributeError(f'factor {name!r} not found')

    def get_variable(self, name):
        if self._variable_dict is None:
            self._set_variable_dict()
        try:
            return self._variable_dict[name]
        except KeyError:
            raise AttributeError(f'variable {name!r} not found')

    def _set_factor_dict(self):
        self._factor_dict = {factor.name: factor for factor in self._factors}

    def _set_variable_dict(self):
        self._variable_dict = {variable.name: variable for variable in self._variables}
