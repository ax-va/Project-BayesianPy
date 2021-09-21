import copy

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class FactoredAlgorithm:
    def __init__(self, model: FactorGraph, query=None, evidence=None):
        # Specifying the model
        self._set_model(model)
        # Specifying the query
        if query is not None:
            self.set_query(query)
        else:
            self._query = None
        # Specifying the evidence
        if evidence is not None:
            self.set_evidence(*evidence)
        else:
            self._evidence = None
        # Probability distribution P(query) or P(query|evidence) of interest
        self._distribution = None

    @property
    def evidence(self):
        return self._evidence

    @property
    def factor_leaves(self):
        return tuple(factor for factor in self._factors if factor.is_leaf())

    @property
    def query(self):
        return self._query

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())

    def has_query_only_one_variable(self):
        if len(self._query) != 1:
            raise ValueError('query contains more than one variable')

    def is_query_set(self):
        # Is a query specified?
        if self._query is None:
            raise AttributeError('query not specified')

    def set_evidence(self, *evidence):
        # Refresh the domain of variables
        self._refresh_algorithm_variables_domain()
        if not evidence[0]:
            self._evidence = None
        else:
            self._set_evidence(*evidence)
    
    def set_query(self, *variables):
        self._query = []
        for query_var in variables:
            # Variable 'query' of interest for computing P(query) or P(query|evidence)
            try:
                query_var = self._get_algorithm_variable(query_var)
            except ValueError:
                self._query = None
                raise ValueError(f'no variable in the model that corresponds to query variable {query_var.name!r}')
            self._check_query_variable_in_evidence(query_var)
            self._query.append(query_var)
        self._query = tuple(sorted(self._query, key=lambda x: x.name))

    def _check_evidence_variable_domain(self, ev_var, ev_val):
        if ev_val not in ev_var.domain:
            self._evidence = None
            raise ValueError(f'value {ev_val!r} is not in the domain {ev_var.domain} of variable {ev_var.name!r}')

    def _check_evidence_variable_in_query(self, ev_var):
        if self._query is not None:
            if ev_var in self._query:
                self._evidence = None
                raise ValueError(f'evidence variable {ev_var.name!r} is in query {tuple(q.name for q in self._query)}')

    def _check_query_variable_in_evidence(self, query_var):
        if self._evidence is not None:
            if query_var in (e[0] for e in self._evidence):
                self._query = None
                raise ValueError(f'query variable {query_var.name!r} is in evidence '
                                 f'{tuple((e[0].name, e[1]) for e in self._evidence)}')

    def _create_algorithm_factors_and_variables(self):
        # Encapsulate the factors and variables inside the algorithm
        # Deeply copy the variables
        self._variables = tuple(copy.deepcopy(self._model.variables))
        # Unlink the factors from the variables
        for variable in self._variables:
            variable.unlink_factors()
        # Create new factors
        self._factors = tuple(
            Factor(
                variables=self._create_algorithm_factor_variables(model_factor),
                function=copy.deepcopy(model_factor.function),
                name=copy.deepcopy(model_factor.name)
            ) for model_factor in self._model.factors
        )

    def _create_algorithm_factor_variables(self, model_factor):
        factor_variables = []
        for model_factor_variable in model_factor.variables:
            index = self._model.variables.index(model_factor_variable)
            factor_variables.append(self._variables[index])
        return tuple(factor_variables)

    def _get_algorithm_variable(self, variable):
        # Make sure that the encapsulated variable is got
        return self._variables[self._model.variables.index(variable)]

    def _refresh_algorithm_variables_domain(self):
        # Refresh the domain of variables
        for alg_var, mod_var in zip(self._variables, self._model.variables):
            alg_var.set_domain(mod_var.domain)

    def _set_evidence(self, *evidence):
        self._evidence = []
        # Setting the evidence is equivalent to reducing the domain of the variable to only one value
        for ev_var, ev_val in evidence:
            try:
                ev_var = self._get_algorithm_variable(ev_var)
            except ValueError:
                self._evidence = None
                raise ValueError(f'no variable in the model that corresponds to evidence variable {ev_var.name!r}')
            self._check_evidence_variable_in_query(ev_var)
            self._check_evidence_variable_domain(ev_var, ev_val)
            # Set the new domain containing only one value
            ev_var.set_domain({ev_val})
            self._evidence.append((ev_var, ev_val))
        self._evidence = tuple(sorted(self._evidence, key=lambda x: x[0].name))

    def _set_model(self, model: FactorGraph):
        # Save the model
        self._model = model
        # Encapsulate the factors and variables inside the algorithm.
        # Create self._factors and self._variables.
        self._create_algorithm_factors_and_variables()