class Variable:
    def __init__(self, domain, name):
        self._domain = domain
        self._name = name

    def __str__(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name
