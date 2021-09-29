class FactorGraph:
    def __init__(self, factors):
        self._factors = tuple(sorted(set(factors), key=lambda f: f.name))
        self._variables = tuple(
            sorted(
                set(variable
                    for factor in self._factors
                    for variable in factor.variables),
                key=lambda v: v.name
            )
        )

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables

    def create_factor_cache(self):
        return {factor.variables: factor for factor in self._factors}
    
    def get_factor(self, name):
        for factor in self.factors:
            if factor.name == name:
                return factor
        raise AttributeError(f'the factor with name {name!r} not found')                            

    def get_variable(self, name):
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise AttributeError(f'the variable with name {name!r} not found')
