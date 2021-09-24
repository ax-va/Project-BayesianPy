class Factorization:
    def __init__(self, factors, variables):
        self._factors = tuple(sorted(set(factors), key=lambda f: f.name))
        self._variables = tuple(sorted(set(variables), key=lambda v: v.name))
        self._cache = {factor.variables: factor for factor in self._factors}

    @property
    def cache(self):
        return self._cache

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables
