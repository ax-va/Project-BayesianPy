from pyb4ml.modeling.factor_graph.factorization import Factorization


class FactorGraph:
    def __init__(self, factors):
        # Factorization
        self._factorization = Factorization(factors)

    @property
    def factorization(self):
        return self._factorization

    @property
    def factors(self):
        return self._factorization.factors

    @property
    def variables(self):
        return self._factorization.variables

    def create_factor_cache(self):
        return self._factorization.create_factor_cache()
    
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
