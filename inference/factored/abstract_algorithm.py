import copy
import itertools

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.model import FactorGraph


class FactoredAlgorithm:
    """
    This is an abstract class of a factored algorithm that real factored algorithms
    inherit
    """
    def __init__(self, model: FactorGraph):
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

    @property
    def evidence(self):
        """
        Returns the evidence as the attribute of the algorithm
        """
        return self._evidence

    @property
    def factors(self):
        return self._factor_graph.factors

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
        algorithm.set_evidence((difficulty, 'd0'), (intelligence, 'i1')) sets the
        evidential values 'd0' and 'i1' to the random variables Difficulty and
        Intelligence, respectively.

        In fact, the domain of a variable is reduced to one evidential value.
        The variable is encapsulated in the algorithm and does not change the
        corresponding model variable.
        """
        # Refresh the domain of evidential variables
        self._refresh_evidential_variables_domain_if_necessary()
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

    def _check_evidence_variable_domain(self, ev_var, ev_val):
        if ev_val not in ev_var.domain:
            self._evidence = None
            raise ValueError(f'value {ev_val!r} is not in the domain {ev_var.domain} of variable {ev_var.name!r}')

    def _check_query_and_evidence(self):
        for query_var in self._query:
            self._check_query_variable_in_evidence(query_var)

    def _check_query_variable_in_evidence(self, query_var):
        if self._evidence is not None:
            if query_var in (e[0] for e in self._evidence):
                self._query = None
                raise ValueError(f'query variable {query_var.name!r} is in evidence '
                                 f'{tuple((e[0].name, e[1]) for e in self._evidence)}')

    def _create_algorithm_factor_graph(self):
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
            print()
            print(f'{self._name} stopped')
            print('*' * 40)

    def _refresh_evidential_variables_domain_if_necessary(self):
        # Refresh the domains of evidential variables
        if self._evidence is not None:
            for ev_var, _ in self._evidence:
                ev_var.set_domain(self._model.variables[self.variables.index(ev_var)].domain)

    def _set_evidence(self, *evidence):
        # Check whether the query has duplicates
        if len(evidence) != len(set(evidence)):
            raise ValueError(f'The evidence must not contain duplicates')
        self._evidence = []
        # Setting the evidence is equivalent to reducing the domain of the variable to only one value
        for ev_var, ev_val in evidence:
            try:
                ev_var = self._get_algorithm_variable(ev_var)
            except ValueError:
                self._evidence = None
                raise ValueError(f'no model variable corresponding to evidence variable {ev_var.name!r}')
            self._check_evidence_variable_domain(ev_var, ev_val)
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
                raise ValueError(f'no model variable corresponding to query variable {query_var.name!r}')
            self._query.append(query_var)
        self._query = tuple(sorted(self._query, key=lambda x: x.name))

    def _set_model(self, model: FactorGraph):
        # Save the model
        self._model = model
        # Encapsulate the factors and variables inside the algorithm by using factorization.
        # Create self._factorization.
        self._create_algorithm_factor_graph()