class Bucket:
    def __init__(self):
        self._content = []

    @property
    def content(self):
        return self._content

    def add(self, factor):
        self._content.append(factor)

                          

