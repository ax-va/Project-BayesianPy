import copy

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
        self._query = ()
        # Evidence not specified
        self._evidence = {}
        # Probability distribution P(query) or P(query|evidence) of interest
        self._distribution = None

    @property
    def eliminating_variables(self):
        """
        Returns non-query and non-evidential variables
        """
        if self._query:
            if self._evidence:
                return tuple(variable for variable in self.variables
                             if variable not in self._query and variable not in self._evidence.keys())
            else:
                return tuple(variable for variable in self.variables if variable not in self._query)
        else:
            if self._evidence:
                return tuple(variable for variable in self.variables if variable not in self._evidence.keys())
            else:
                return self.variables

    @property
    def evidence(self):
        """
        Returns the evidence as the attribute of the algorithm
        """
        return self._evidence

    @property
    def factors(self):
        return self._inner_model.factors

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
                        f'the number {len(values)} of given values does not match '
                        f'the number {len(self._query)} of query variables'
                    )
                for variable, value in zip(self._query, values):
                    if value not in variable.domain:
                        raise ValueError(f'value {value!r} not in domain {variable.domain} of {variable.name}')
                return self._distribution[values]
            return distribution
        else:
            raise AttributeError('distribution not computed')

    @property
    def query(self):
        return self._query

    @property
    def variables(self):
        return self._inner_model.variables

    def print_pd(self):
        """
        Prints the complete probability distribution of the query variables
        """
        if self._distribution is not None:
            evidence_str = ' | ' + ', '.join(f'{var.name} = {val!r}' for var, val in self._evidence.items()) \
                if self._evidence \
                else ''
            for values in Variable.evaluate_variables(self._query):
                query_str = 'P(' + ', '.join(f'{var.name} = {val!r}' for var, val in zip(self._query, values))
                value_str = str(self.pd(*values))
                equal_str = ') = '
                print(query_str + evidence_str + equal_str + value_str)
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
        # ...
        self._delete_evidence()
        if evidence[0]:
            self._set_evidence(*evidence)
        else:
            self._evidence = {}
    
    def set_query(self, *variables):
        """
        Sets the query. For example, algorithm.set_query(difficulty, intelligence)
        sets the random variables Difficulty and Intelligence as the query. The values of
        variables in a computed probability distribution must have the same order. For example,
        algorithm.pd('d0', 'i1') returns a probability corresponding to Difficulty = 'd0' and
        Intelligence = 'i1'.
        """
        if variables[0]:
            self._set_query(*variables)
        else:
            self._query = ()

    def _check_query_and_evidence(self):
        if self._evidence:
            query_set = set(self._query)
            evidence_set = set(self._evidence.keys())
            if query_set.intersection(evidence_set) != set():
                raise ValueError(f'query variables {set(query_var.name for query_var in query_set)} '
                                 f'and evidential variables {set(ev_var.name for ev_var in evidence_set)} '
                                 f'must be disjoint')

    def _delete_evidence(self):
        for var in self._evidence.keys():
            var.set_domain(self._inner_to_outer_variables[var].domain)
        del self._evidence
        self._evidence = {}

    def _has_query_only_one_variable(self):
        if len(self._query) > 1:
            raise ValueError('the query contains more than one variable')
        if len(self._query) < 1:
            raise ValueError('the query contains less than one variable')

    def _is_query_set(self):
        if not self._query:
            raise AttributeError('query not specified')

    def _print_start(self):
        if self._print_info:
            print('*' * 40)
            print(f'{self._name} started')

    def _print_stop(self):
        if self._print_info:
            print(f'\n{self._name} stopped')
            print('*' * 40)

    def _set_evidence(self, *evidence):
        ev_variables = tuple(var_val[0] for var_val in evidence)
        if len(ev_variables) != len(set(ev_variables)):
            raise ValueError(f'the evidence must not contain duplicates')
        for outer_var, val in evidence:
            try:
                inner_var = self._outer_to_inner_variables[outer_var]
            except KeyError:
                self._evidence = {}
                raise KeyError(f'no model variable corresponds to evidence variable {outer_var.name}')
            try:
                inner_var.check_value(val)
            except ValueError as exception:
                self._evidence = {}
                raise exception
            # Set the new domain containing only one value
            inner_var.set_domain({val})
            self._evidence[inner_var] = val

    def _set_query(self, *variables):
        # Check whether the query has duplicates
        if len(variables) != len(set(variables)):
            raise ValueError(f'query must not contain duplicates')
        try:
            self._query = tuple(
                sorted(
                    (self._outer_to_inner_variables[outer_var] for outer_var in variables),
                    key=lambda x: x.name
                )
            )
        except KeyError:
            self._query = ()
            raise ValueError(f'some query variables are not model variables')
        except Exception:
            raise ValueError(f'some query variables are incorrect')

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