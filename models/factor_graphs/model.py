from pyb4ml.modeling.factor_graph.factorization import Factorization


class Model:
    def __init__(self, factors, variables):
        # Factorization
        self._factorization = Factorization(
            factors=factors,
            variables=variables
        )

    @property
    def factorization(self):
        return self._factorization

    @property
    def factors(self):
        return self._factorization.factors

    @property
    def variables(self):
        return self._factorization.variables

    def get_variable(self, name):
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise ValueError('the variable with name {name!r} not found')