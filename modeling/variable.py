class Variable:
    def __init__(self, domain, name=None):
        self._domain = domain
        if name is None:
            self._name = str(id(self))
        else:
            self._name = name

    def __str__(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name
