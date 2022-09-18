class ModelElement:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    @property
    def name(self):
        return self._name
