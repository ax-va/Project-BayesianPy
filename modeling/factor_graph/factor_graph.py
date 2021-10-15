class FactorGraph:
    def __init__(self, factors):
        self._set_attributes(factors)

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

    def extend_factors(self, by_factors):
        for variables in self.factor_variables:
            for factor in by_factors:
                if set(variables) == set(factor.variables):
                    raise ValueError(f'the model cannot be extended by factor {factor.name}')
        self._factors.extend(by_factors)
        self._set_attributes(self._factors)
    
    def get_factor(self, name):
        try:
            return self._factor_dict[name]
        except KeyError:
            raise AttributeError(f'factor {name!r} not found')

    def get_variable(self, name):
        try:
            return self._variable_dict[name]
        except KeyError:
            raise AttributeError(f'variable {name!r} not found')

    def replace_factors(self, with_names, by_factors):
        for name, factor in zip(with_names, by_factors):
            try:
                self._factor_dict[name] = factor
            except KeyError:
                self._factor_dict = {factor.name: factor for factor in self._factors}
                raise AttributeError(f'factor {name!r} not found')
        self._set_attributes(self._factor_dict.values())

    def _set_attributes(self, factors):
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
        self._factor_dict = {factor.name: factor for factor in self._factors}
        self._variable_dict = {variable.name: variable for variable in self._variables}
