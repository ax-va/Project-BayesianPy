from pyb4ml.inference.factored.factored_algorithm import FactoredAlgorithm


def _get_cost(variable):
    cost = 0
    var_neighbors = variable.neighbors
    length = len(var_neighbors)
    for i1 in range(length - 1):
        for i2 in range(i1 + 1, length):
            if var_neighbors[i1] not in var_neighbors[i2].neighbors:
                cost += 1
    return cost


def _get_weighted_cost(variable):
    cost = 0
    var_neighbors = variable.neighbors
    length = len(var_neighbors)
    for i1 in range(length - 1):
        for i2 in range(i1 + 1, length):
            if var_neighbors[i1] not in var_neighbors[i2].neighbors:
                cost += len(var_neighbors[i1].domain) * len(var_neighbors[i2].domain)
    return cost


class GOA(FactoredAlgorithm):
    """
    Greedy ordering algorithm
    """
    _cost_functions = {
        'min-fill': _get_cost,
        'weighted-min-fill': _get_weighted_cost
    }

    def __init__(self, model):
        FactoredAlgorithm.__init__(self, model)
        self._ordering = None
        self._not_ordered = None
        self._cost_function = None
        self._cost = None

    @staticmethod
    def _link_neighbors(variable):
        var_neighbors = variable.neighbors
        for i in range(len(var_neighbors)):
            var_neighbor = var_neighbors[i]
            var_neighbor.neighbors.extend([var for var in var_neighbors if var is not var_neighbor])
            var_neighbor.neighbors = list(set(var_neighbors))

    @property
    def ordering(self):
        return self._ordering

    def run(self, cost='weighted-min-fill'):
        self._cost = cost
        self._cost_function = GOA._cost_functions[self._cost]
        self._set_neighbors()
        self._ordering = []
        self._not_ordered = list(variable for variable in self.non_query_variables())
        while len(self._not_ordered) > 0:
            elm_var = self._eliminate_variable()
            self._ordering.append(elm_var)
            GOA._link_neighbors(elm_var)

    def _eliminate_variable(self):
        min_variable = self._not_ordered[0]
        min_cost_val = self._cost_function(min_variable)
        min_index = -1
        for index, variable in enumerate(self._not_ordered[1:len(self._not_ordered)]):
            cost_val = self._cost_function(variable)
            if cost_val < min_cost_val:
                min_cost_val = cost_val
                min_variable = variable
                min_index = index
        del self._not_ordered[min_index + 1]
        return min_variable

    def _set_neighbors(self):
        for variable in self.variables:
            variable.neighbors = sorted(
                set(var for factor in variable.factors for var in factor.variables if var is not variable),
                key=lambda f: f.name
            )
