"""
The module contains the class of the Greedy Ordering algorithm.

Attention:  The author is not responsible for any damage that can be caused by the use
of this code.  You use this code at your own risk.  Any claim against the author is
legally void.  By using this code, you agree to the terms imposed by the author.

Achtung:  Der Autor haftet nicht für Schäden, die durch die Verwendung dieses Codes
entstehen können.  Sie verwenden dieses Code auf eigene Gefahr.  Jegliche Ansprüche
gegen den Autor sind rechtlich nichtig.  Durch die Verwendung dieses Codes stimmen
Sie den vom Autor auferlegten Bedingungen zu.

© 2021 Alexander Vasiliev
"""
from pyb4ml.inference.factored.factored_algorithm import FactoredAlgorithm


class GO(FactoredAlgorithm):
    """
    This implementation of the Greedy Ordering (GO) algorithm finds a near-optimal
    variable elimination order that can be used later, for example, in the BE
    algorithm.  The GO algorithm uses greedy search, here, with the cost criterion
    of "min-fill" or "weighted-min-fill".  Eliminating variables leads to the
    appearance of new factors.  That can be graphically represented as edges already
    existing or needed to be added between all neighbors of an eliminated node in
    a (moralized in the case of directed edges) graph.  The best order implies
    that each bucket should have a cardinality of its free variables as small as 
    possible so that a whole number of computational operations is also as small as
    possible.  One way to build an elimination order close to the best one is
    to greedily remove the variables such that there are the additional edges to be 
    added as few as possible.  That corresponds to the cost criterion of "min-fill".
    If the weights of additional edges, i.e. the products of the cardinality 
    of edge variables, are taken into account, such a cost criterion is called 
    "weighted-min-fill".  Those two heuristic approaches often work surprisingly well
    in practice.  See, for example, [KF09] for more details.

    Here, the query and evidence are optional.  The GO algorithm returns an elimination
    order as a tuple of variables, in which the first variable will be eliminated
    first, the second variable second, and so on.

    Restrictions:  Only works with random variables with categorical value domains.

    Recommended:  In the case of trees, the BP algorithm automatically finds the best
    elimination order.  Therefore, in that case, it is recommended to use the BP
    algorithm instead of the bundle of the GO and BE algorithms.

    References:

    [KF09] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles
    and Techniques", The MIT Press, 2009
    """
    _name = 'Greedy Ordering'

    def __init__(self, model):
        FactoredAlgorithm.__init__(self, model)
        self._order_number = None
        self._elimination_order = []
        self._not_ordered_variables = []
        self._cost_function = None
        self._cost = None
        self._print_info = None
        self._cost_functions = {
            'min-fill': self._get_fill_cost,
            'weighted-min-fill': self._get_weighted_fill_cost
        }

    @staticmethod
    def _link_neighbors(variable):
        var_neighbors = variable.neighbors
        for i in range(len(var_neighbors)):
            var_neighbor = var_neighbors[i]
            var_neighbor.neighbors.extend(var for var in var_neighbors if var is not var_neighbor)
            var_neighbor.neighbors = list(set(var_neighbor.neighbors))

    @property
    def order(self):
        return tuple(self._inner_to_outer_variables[var] for var in self._elimination_order)

    def print_order(self):
        self.print_query()
        self.print_evidence()
        print('Elimination order: ' + ', '.join(variable.name for variable in self._elimination_order))

    def run(self, cost='weighted-min-fill', print_info=False):
        self._print_info = print_info
        self._order_number = 0
        self._cost = cost
        self._cost_function = self._cost_functions[self._cost]
        self._elimination_order = []
        self._not_ordered_variables = list(variable for variable in self.elimination_variables)
        self._set_neighbors()
        self._print_start()
        while len(self._not_ordered_variables) > 0:
            self._print_candidates()
            elm_var = self._eliminate_min_cost_variable()
            self._elimination_order.append(elm_var)
            GO._link_neighbors(elm_var)
        self._print_stop()

    def _eliminate_min_cost_variable(self):
        min_variable = self._not_ordered_variables[0]
        min_cost_val = self._cost_function(min_variable)
        self._print_total_cost(min_cost_val, min_variable)
        min_index = -1
        for index, variable in enumerate(self._not_ordered_variables[1:len(self._not_ordered_variables)]):
            cost_val = self._cost_function(variable)
            self._print_total_cost(cost_val, variable)
            if cost_val < min_cost_val:
                min_cost_val = cost_val
                min_variable = variable
                min_index = index
        self._print_before_elimination(min_variable)
        del self._not_ordered_variables[min_index + 1]
        for neighbor in min_variable.neighbors:
            neighbor.neighbors.remove(min_variable)
        self._print_after_elimination(min_variable)
        return min_variable

    def _get_fill_cost(self, variable):
        cost_sum = 0
        var_neighbors = variable.neighbors
        length = len(var_neighbors)
        for i1 in range(length - 1):
            for i2 in range(i1 + 1, length):
                neighbor1 = var_neighbors[i1]
                neighbor2 = var_neighbors[i2]
                if neighbor1 not in neighbor2.neighbors:
                    cost_sum += 1
                    self._print_fill_cost(neighbor1, neighbor2, 1)
        return cost_sum

    def _get_weighted_fill_cost(self, variable):
        cost_sum = 0
        var_neighbors = variable.neighbors
        length = len(var_neighbors)
        for i1 in range(length - 1):
            for i2 in range(i1 + 1, length):
                neighbor1 = var_neighbors[i1]
                neighbor2 = var_neighbors[i2]
                if neighbor1 not in neighbor2.neighbors:
                    cost = len(neighbor1.domain) * len(neighbor2.domain)
                    cost_sum += cost
                    self._print_fill_cost(neighbor1, neighbor2, cost)
        return cost_sum

    def _print_after_elimination(self, variable):
        if self._print_info:
            print('\nAfter the elimination of the variable:')
            for neighbor in variable.neighbors:
                print('-- ' + variable.name + "'s neighbor: " + neighbor.name)
                for var in neighbor.neighbors:
                    print('---- ' + neighbor.name + "'s neighbor: " + var.name)

    def _print_before_elimination(self, variable):
        if self._print_info:
            print(str(self._order_number) + ': ' + variable.name)
            self._order_number += 1
            print('\nBefore the elimination of the variable:')
            for neighbor in variable.neighbors:
                print('-- ' + variable.name + "'s neighbor: " + neighbor.name)
                for var in neighbor.neighbors:
                    print('---- ' + neighbor.name + "'s neighbor: " + var.name)

    def _print_candidates(self):
        if self._print_info:
            print('\nVariable candidates to be eliminated:\n')

    def _print_fill_cost(self, neighbor1, neighbor2, cost):
        if self._print_info:
            print(f'cost({neighbor1.name} - {neighbor2.name}) = {cost}')

    def _print_total_cost(self, cost, variable):
        if self._print_info:
            print(f'total_cost({variable.name}) = {cost}\n')

    def _set_neighbors(self):
        for variable in self.non_evidential:
            variable.neighbors = list(
                set(var
                    for factor in variable.factors
                    for var in factor.variables
                    if var is not variable and not var.is_evidential()
                    )
            )
