class Message:
    def __init__(self, from_node, to_node, values):
        self._from_node = from_node
        self._to_node = to_node
        self._values = values

    def __call__(self, value):
        return self._values[value]

    @property
    def from_node(self):
        return self._from_node

    @property
    def to_node(self):
        return self._to_node

    @property
    def values(self):
        return self.values


class Messages:
    def __init__(self):
        self._messages = {}

    def __contains__(self, message):
        return (message.from_node, message.to_node) in self._messages

    def __iter__(self):
        return iter(self._messages)

    def add(self, message):
        self._messages[(message.from_node, message.to_node)] = message

    def contains(self, from_node, to_node):
        return (from_node, to_node) in self._messages.keys()

    def get(self, from_node, to_node):
        return self._messages[(from_node, to_node)]



