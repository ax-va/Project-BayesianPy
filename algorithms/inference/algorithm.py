import copy

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.variable import Variable
from pyb4ml.models.factor_graphs.model import Model


class InferenceAlgorithm:
    def __init__(self, model: Model, query=None, evidence=None):
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
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())

    def has_query_only_one_variable(self):
        if len(self._query) != 1:
            raise ValueError('query has more than one variables')

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
            self._evidence = []
            # Setting the evidence is equivalent to reducing the domain of the variable to only one value
            for var, val in evidence:
                if var is self._query:
                    self._evidence = None
                    raise ValueError(f'evidence variable {var} and query variable {self._query} must not match')
                if val not in var.domain:
                    self._evidence = None
                    raise ValueError(f'value {val} is not in the domain {var.domain} of variable {var}')
                try:
                    evidence_var = self._get_algorithm_variable(var)
                except ValueError:
                    self._evidence = None
                    raise ValueError(f'no variable in the model that corresponds to evidence variable {var}')
                # Set the new domain containing only one value
                evidence_var.set_domain({val})
                self._evidence.append((evidence_var, val))
            self._evidence = tuple(sorted(self._evidence, key=lambda x: x[0].name))
    
    def set_query(self, *variables):
        self._query = []
        for variable in variables:
            # Variable 'query' of interest for computing P(query) or P(query|evidence)
            try:
                query = self._get_algorithm_variable(variable)
            except ValueError:
                raise ValueError(f'no variable in the model that corresponds to query variable {variable}')
            self._query.append(query)
        self._query = tuple(sorted(self._query, key=lambda x: x.name))

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

    def _set_model(self, model: Model):
        # Save the model
        self._model = model
        # Encapsulate the factors and variables inside the algorithm.
        # Create self._factors and self._variables.
        self._create_algorithm_factors_and_variables()