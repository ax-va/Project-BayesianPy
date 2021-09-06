class Variable:
    def __init__(self, domain):
        self._domain = domain

    @property
    def domain(self):
        return self._domain
