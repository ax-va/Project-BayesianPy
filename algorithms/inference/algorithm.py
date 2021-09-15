import copy

from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.variable import Variable
from pyb4ml.models.factor_graphs.model import Model


class InferenceAlgorithm:
    def __init__(self, model: Model = None, query: Variable = None, evidence=None):
        # Specifying the model
        if model is not None:
            self.set_model(model)
        else:
            self._model = None
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
    def pd(self):
        """
        Returns the probability distribution P(Q) or if an evidence is set then
        P(Q|E_1 = e_1, ..., E_k = e_k) as a function of q, where q is in the domain
        of random variable Q
        """
        if self._distribution is not None:
            def distribution(value):
                if value not in self._query.domain:
                    raise ValueError(f'the value {value!r} is not in the domain {self._query.domain}')
                return self._distribution[value]
            return distribution
        else:
            raise AttributeError('distribution not computed')

    @property
    def variable_leaves(self):
        return tuple(variable for variable in self._variables if variable.is_non_isolated_leaf())

    def is_model_set(self):
        # Is a model specified?
        if self._model is None:
            raise AttributeError('model not specified')

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
                    raise ValueError(f'there is no variable in the model that corresponds to evidence variable {var}')
                # Set the new domain containing only one value
                evidence_var.set_domain({val})
                self._evidence.append((evidence_var, val))
            self._evidence = tuple(sorted(self._evidence, key=lambda x: x[0].name))

    def set_model(self, model: Model):
        # Save the model
        self._model = model
        # Encapsulate the factors and variables inside the algorithm.
        # Create self._factors and self._variables.
        self._create_algorithm_factors_and_variables()
    
    def set_query(self, variable: Variable):
        # Variable 'query' of interest for computing P(query) or P(query|evidence)
        try:
            self._query = self._get_algorithm_variable(variable)
        except ValueError:
            raise ValueError(f'there is no variable in the model that corresponds to query variable {variable}')

    def print_pd(self):
        if self._distribution is not None:
            if self._evidence is None:
                for value in self._query.domain:
                    print(f'P({self._query}={value!r})={self.pd(value)}')
            else:
                ev_str = '|' + ', '.join(f'{ev_var.name}={ev_val!r}' for ev_var, ev_val in self._evidence) + ')'
                for value in self._query.domain:
                    print(f'P({self._query}={value!r}{ev_str}={self.pd(value)}')
        else:
            raise AttributeError('distribution not computed')

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