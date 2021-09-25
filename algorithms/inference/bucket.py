class Bucket:
    def __init__(self, variable):
        self._variable = variable
        self._content = []

    @property
    def content(self):
        return self._content

    @property
    def variable(self):
        return self._variable

    def add(self, factor):
        self._content.append(factor)
