def _min_fill(variable):
    cost = 0
    for neighbor1 in variable.neighbors:
        for neighbor2 in variable.neighbors:
            if neighbor1 not in neighbor2.neighbors:
                cost += 1
    return cost


def _weighted_min_fill(variable):
    pass


class GOA:
    _cost_functions = {
        'min-fill': _min_fill,
        'weighted-min-fill': _weighted_min_fill
    }

    def __init__(self, variables):
        self._variables = variables
        self._query = None
        self._non_query_variables = None
        self._ordering = None
        self._not_ordered = None
        self._cost_function = None
        self._cost = None

    def run(self, cost='weighted-min-fill'):
        self._cost = cost
        cost_function = GOA._cost_functions[self._cost]
        self._set_neighbors()
        self._non_query_variables = tuple(variable for variable in self._variables if variable not in self._query)
        self._ordering = []
        self._not_ordered = {variable: cost_function(variable) for variable in self._non_query_variables}
        while len(self._not_ordered) > 0:
            elimination_variable = self._select_variable

    def set_query(self, query):
        self._query = query

    def _set_neighbors(self):
        for variable in self._variables:
            variable.neighbors = sorted(
                set(var for factor in variable.factors for var in factor.variables if var is not variable),
                key=lambda f: f.name
            )
