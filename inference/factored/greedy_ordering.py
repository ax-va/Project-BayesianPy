from pyb4ml.inference.factored.factored_algorithm import FactoredAlgorithm


def _get_fill_cost(variable):
    cost_sum = 0
    var_neighbors = variable.neighbors
    length = len(var_neighbors)
    for i1 in range(length - 1):
        for i2 in range(i1 + 1, length):
            if var_neighbors[i1] not in var_neighbors[i2].neighbors:
                cost_sum += 1
    return cost_sum


def _get_weighted_fill_cost(variable):
    cost_sum = 0
    var_neighbors = variable.neighbors
    length = len(var_neighbors)
    for i1 in range(length - 1):
        for i2 in range(i1 + 1, length):
            if var_neighbors[i1] not in var_neighbors[i2].neighbors:
                cost_sum += len(var_neighbors[i1].domain) * len(var_neighbors[i2].domain)
    return cost_sum


class GO(FactoredAlgorithm):
    """
    Greedy ordering
    """
    _cost_functions = {
        'min-fill': _get_fill_cost,
        'weighted-min-fill': _get_weighted_fill_cost
    }

    def __init__(self, model):
        FactoredAlgorithm.__init__(self, model)
        self._order = None
        self._ordering = None
        self._not_ordered = None
        self._cost_function = None
        self._cost = None
        self._print_info = None
        self._name = 'Greedy Ordering'

    @staticmethod
    def _link_neighbors(variable):
        var_neighbors = variable.neighbors
        for i in range(len(var_neighbors)):
            var_neighbor = var_neighbors[i]
            var_neighbor.neighbors.extend([var for var in var_neighbors if var is not var_neighbor])
            var_neighbor.neighbors = list(set(var_neighbor.neighbors))

    @property
    def ordering(self):
        return tuple(self._model.get_variable(variable.name) for variable in self._ordering)

    def print_ordering(self):
        if self._query is not None:
            print('Query: ' + ', '.join(variable.name for variable in self.query))
        else:
            print('No query')
        if self._evidence is not None:
            print('Evidence: ' + ', '.join(f'{ev_var.name} = {ev_val!r}' for ev_var, ev_val in self._evidence))
        else:
            print('No evidence')
        print('Elimination ordering: ' + ', '.join(variable.name for variable in self._ordering))

    def run(self, cost='weighted-min-fill', print_info=False):
        self._print_info = print_info
        self._order = 0
        self._cost = cost
        self._cost_function = GO._cost_functions[self._cost]
        self._ordering = []
        self._not_ordered = list(variable for variable in self.non_query_variables)
        self._set_neighbors()
        self._print_start()
        while len(self._not_ordered) > 0:
            elm_var = self._eliminate_min_cost_variable()
            self._ordering.append(elm_var)
            GO._link_neighbors(elm_var)
        self._print_stop()

    def _eliminate_min_cost_variable(self):
        min_variable = self._not_ordered[0]
        min_cost_val = self._cost_function(min_variable)
        self._print_first_cost(min_cost_val, min_variable)
        min_index = -1
        for index, variable in enumerate(self._not_ordered[1:len(self._not_ordered)]):
            cost_val = self._cost_function(variable)
            self._print_cost(cost_val, variable)
            if cost_val < min_cost_val:
                min_cost_val = cost_val
                min_variable = variable
                min_index = index
        self._print_before_elimination(min_variable)
        del self._not_ordered[min_index + 1]
        for neighbor in min_variable.neighbors:
            neighbor.neighbors.remove(min_variable)
        self._print_after_elimination(min_variable)
        return min_variable

    def _print_after_elimination(self, variable):
        if self._print_info:
            print('After the elimination of the variable:')
            for neighbor in variable.neighbors:
                print('-- ' + variable.name + "'s neighbor: " + neighbor.name)
                for var in neighbor.neighbors:
                    print('---- ' + neighbor.name + "'s neighbor: " + var.name)

    def _print_before_elimination(self, variable):
        if self._print_info:
            print('\n' + str(self._order) + ': ' + variable.name)
            self._order += 1
            print('Before the elimination of the variable:')
            for neighbor in variable.neighbors:
                print('-- ' + variable.name + "'s neighbor: " + neighbor.name)
                for var in neighbor.neighbors:
                    print('---- ' + neighbor.name + "'s neighbor: " + var.name)

    def _print_cost(self, cost, variable):
        if self._print_info:
            print(f'cost({variable.name}) = {cost}')

    def _print_first_cost(self, cost, variable):
        if self._print_info:
            print(f'\ncost({variable.name}) = {cost}')

    def _set_neighbors(self):
        for variable in self.variables:
            variable.neighbors = list(
                set(var for factor in variable.factors for var in factor.variables if var is not variable)
            )
