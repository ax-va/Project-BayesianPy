class Model:
    @property
    def factorization(self):
        return self._factorization

    @property
    def factors(self):
        return self._factorization.factors

    @property
    def variables(self):
        return self._variables