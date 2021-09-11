from pyb4ml.modeling.factor_graph.factorization import Factorization


class Model:
    def __init__(self, factors, variables):
        # factorization
        self._factorization = Factorization(
            factors=factors,
            variables=variables
        )
        self._factors = self._factorization.factors
        self._variables = self._factorization.variables

    @property
    def factorization(self):
        return self._factorization

    @property
    def factors(self):
        return self._factors

    @property
    def variables(self):
        return self._variables