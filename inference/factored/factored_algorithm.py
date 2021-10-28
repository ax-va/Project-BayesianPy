import copy
import itertools

from pyb4ml.modeling.categorical.variable import Variable
from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class FactoredAlgorithm:
    """
    This is an abstract class of a factored algorithm that real factored algorithms
    inherit
    """
    def __init__(self, model: FactorGraph):
        # Inner model not specified
        self._inner_model = None
        # Outer model not specified
        self._outer_model = None
        # Specify the model, sets self._factor_graph
        self._set_model(model)
        # Query not specified
        self._query = None
        # Evidence not specified
        self._evidence = None
        # Probability distribution P(query) or P(query|evidence) of interest
        self._distribution = None

    @staticmethod
    def evaluate_variables(variables):
        domains = (variable.domain for variable in variables)
        return tuple(itertools.product(*domains))

    @staticmethod
    def split_evidential_and_non_evidential_variables(variables, without_variables=()):
        """
        Splits evidential and non-evidential variables ignoring without_variables
        """
        evidential_variables = []
        non_evidential_variables = []
        for variable in variables:
            if variable not in without_variables:
                if variable.is_evidential():
                    evidential_variables.append(variable)
                else:
                    non_evidential_variables.append(variable)
        return tuple(evidential_variables), tuple(non_evidential_variables)

    @property
    def evidence(self):
        """
        Returns the evidence as the attribute of the algorithm
        """
        return self._evidence

    @property
    def factors(self):
        if self._factors is None:
            self._factors = list(self._factor_graph.factors)
        return self._factors

    @property
    def non_query_variables(self):
        return tuple(variable for variable in self.variables if variable not in self._query) \
            if self._query is not None else self.variables

    @property
    def pd(self):
        """
        Returns the probability distribution P(Q_1, ..., Q_s) or if an evidence is set then
        P(Q_1, ..., Q_s | E_1 = e_1, ..., E_k = e_k) as a function of q_1, ..., q_s, where
        q_1, ..., q_s are in the value domains of random variable Q_1, ..., Q_s, respectively.

        The order of values must correspond to the order of variables in the query.  For example,
        if algorithm.set_query(difficulty, intelligence) sets the random variables Difficulty
        and Intelligence as the query, then algorithm.pd('d0', 'i1') returns a probability
        corresponding to Difficulty = 'd0' and Intelligence = 'i1'.
        """
        if self._distribution is not None:
            def distribution(*values):
                if len(values) != len(self._query):
                    raise ValueError(
                        f'The number {len(values)} of values does not match '
                        f'the number {len(self._query)} of variables in the query'
                    )
                for variable, value in zip(self._query, values):
                    if value not in variable.domain:
                        raise ValueError(f'value {value!r} not in domain {variable.domain} of {variable.name!r}')
                return self._distribution[values]
            return distribution
        else:
            raise AttributeError('distribution not computed')

    @property
    def query(self):
        return self._query

    @property
    def variables(self):
        return self._factor_graph.variables

    def print_pd(self):
        """
        Prints the complete probability distribution of the query variables
        """
        if self._distribution is not None:
            evaluated_query = FactoredAlgorithm.evaluate_variables(self._query)
            ev_str = '' \
                if self._evidence is None \
                else ' | ' + ', '.join(f'{ev_var.name} = {ev_val!r}' for ev_var, ev_val in self._evidence)
            for values in evaluated_query:
                query_str = 'P(' + ', '.join(f'{var.name} = {val!r}' for var, val in zip(self._query, values))
                value_str = str(self.pd(*values))
                equal_str = ') = '
                print(query_str + ev_str + equal_str + value_str)
        else:
            raise AttributeError('distribution not computed')

    def set_evidence(self, *evidence):
        """
        Sets the evidence. For example,
        algorithm.set_evidence((difficulty, 'd0'), (intelligence, 'i1')) assigns the
        evidential values 'd0' and 'i1' to the random variables Difficulty and
        Intelligence, respectively.

        In fact, the domain of a variable is reduced to one evidential value.
        The variable is encapsulated in the algorithm and the domain of the 
        corresponding model variable is not changed.
        """
        # Refresh the domain of evidential variables
        self._refresh_evidential_variables()
        if not evidence[0]:
            self._evidence = None
        else:
            self._set_evidence(*evidence)
    
    def set_query(self, *variables):
        """
        Sets the query. For example, algorithm.set_query(difficulty, intelligence)
        sets the random variables Difficulty and Intelligence as the query. The values of
        variables in a computed probability distribution must have the same order. For example,
        algorithm.pd('d0', 'i1') returns a probability corresponding to Difficulty = 'd0' and
        Intelligence = 'i1'.
        """
        if not variables[0]:
            self._query = None
        else:
            self._set_query(*variables)

    def _check_query_and_evidence(self):
        if self._evidence is not None:
            query_set = set(self._query)
            evidence_set = set(ev_var for ev_var, _ in self._evidence)
            if query_set.intersection(evidence_set) != set():
                raise ValueError(f'query variables {set(query_var.name for query_var in query_set)} '
                                 f' and evidential variables {set(ev_var.name for ev_var in evidence_set)}'
                                 f' are not disjoint')

        # Encapsulate the factors and variables inside the algorithm.
        # Deeply copy the variables.
        variables = tuple(copy.deepcopy(self._model.variables))
        # Unlink the factors from the variables
        for variable in variables:
            variable.unlink_factors()
        # Create new factors
        factors = tuple(
            Factor(
                variables=self._create_algorithm_factor_variables(model_factor, variables),
                function=copy.deepcopy(model_factor.function),
                name=copy.deepcopy(model_factor.name)
            ) for model_factor in self._model.factors
        )
        self._factor_graph = FactorGraph(factors)

    def _create_algorithm_factor_variables(self, model_factor, variables):
        return tuple(variables[self._model.variables.index(model_factor_variable)]
                     for model_factor_variable in model_factor.variables)

    def _get_algorithm_variable(self, variable):
        # Make sure that the encapsulated variable is got
        return self.variables[self._model.variables.index(variable)]

    def _has_query_only_one_variable(self):
        if len(self._query) > 1:
            raise ValueError('the query contains more than one variable')

    def _is_query_set(self):
        # Is a query specified?
        if self._query is None:
            raise AttributeError('query not specified')

    def _print_start(self):
        if self._print_info:
            print('*' * 40)
            print(f'{self._name} started')

    def _print_stop(self):
        if self._print_info:
            print(f'\n{self._name} stopped')
            print('*' * 40)

    def _refresh_evidential_variables(self):
        # Refresh the domains of evidential variables and refresh evidential factors
        if self._evidence is not None:
            for ev_var, _ in self._evidence:
                ev_var.set_domain(self._model.variables[self.variables.index(ev_var)].domain)

    def _set_evidence(self, *evidence):

        self._evidence = []
        # Setting the evidence is equivalent to reducing the domain of the variable to only one value
        for ev_var, ev_val in evidence:
            try:
                ev_var = self._get_algorithm_variable(ev_var)
            except ValueError:
                self._evidence = None
                raise ValueError(f'no model variable corresponding to evidence variable {ev_var.name}')
            try:
                ev_var.check_value(ev_val)
            except ValueError as exc:
                self._evidence = None
                raise exc
            # Set the new domain containing only one value
            ev_var.set_domain({ev_val})
            self._evidence.append((ev_var, ev_val))
        self._evidence = tuple(sorted(self._evidence, key=lambda x: x[0].name))

    def _set_query(self, *variables):
        # Check whether the query has duplicates
        if len(variables) != len(set(variables)):
            raise ValueError(f'The query must not contain duplicates')
        self._query = []
        for query_var in variables:
            # Variable 'query' of interest for computing P(query) or P(query|evidence)
            try:
                query_var = self._get_algorithm_variable(query_var)
            except ValueError:
                self._query = None
                raise ValueError(f'no model variable corresponding to query variable {query_var.name}')
            self._query.append(query_var)
        self._query = tuple(sorted(self._query, key=lambda x: x.name))

    def _set_model(self, model: FactorGraph):
        self._outer_model = model
        # Create algorithm variables (inner variables)
        self._inner_to_outer_variables = {}
        self._outer_to_inner_variables = {}
        for outer_variable in self._outer_model.variables:
            inner_variable = Variable(
                domain=outer_variable.domain,
                name=copy.deepcopy(outer_variable.name)
            )
            self._inner_to_outer_variables[inner_variable] = outer_variable
            self._outer_to_inner_variables[outer_variable] = inner_variable
        # Create algorithm factors (inner factors)
        self._inner_to_outer_factors = {}
        self._outer_to_inner_factors = {}
        for outer_factor in self._outer_model.factors:
            inner_factor = Factor(
                variables=tuple(self._outer_to_inner_variables[outer_var] for outer_var in outer_factor.variables),
                function=copy.deepcopy(outer_factor.function),
                name=copy.deepcopy(outer_factor.name)
            )
            self._inner_to_outer_factors[inner_factor] = outer_factor
            self._outer_to_inner_factors[outer_factor] = inner_factor
        # Create an algorithm model (an inner model)
        self._inner_model = FactorGraph(factors=self._inner_to_outer_factors.keys())