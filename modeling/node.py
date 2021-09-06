class Node:
    def __init__(self, name):
        self._name = name if name is not None else str(id(self))

    def __str__(self):
        return self._name

    @property
    def name(self):
        return self._name
