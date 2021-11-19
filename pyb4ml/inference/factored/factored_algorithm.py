import copy

from pyb4ml.modeling.categorical.variable import Variable
from pyb4ml.modeling.factor_graph.factor import Factor
from pyb4ml.modeling.factor_graph.factor_graph import FactorGraph


class FactoredAlgorithm:
    """
    This is an abstract class of some factored algorithm, which is inherited by
    the classes of real factored algorithms, e.g. the Belief Propagation or Bucket
    Elimination algorithms.  The class contains and defines common attributes and 
    methods, respectively. 
    """
    def __init__(self, model: FactorGraph):
        # Inner model not specified
        self._inner_model = None
        # Outer model not specified
        self._outer_model = None
        # Specify the outer and inner models
        self._set_model(model)
        # Query not specified
        self._query = ()
        # Evidence not specified
        self._evidence = ()
        # Evidence tuples not specified
        self._evidence_tuples = ()
        # Probability distribution P(query) or P(query|evidence) not specified
        self._distribution = None

    @property
    def elimination_variables(self):
        """
        Returns non-query and non-evidential variables
        """
        if self._query:
            if self._evidence:
                return tuple(var for var in self.variables if var not in self._query and var not in self._evidence)
            else:
                return tuple(var for var in self.variables if var not in self._query)
        else:
            if self._evidence:
                return tuple(var for var in self.variables if var not in self._evidence)
            else:
                return self.variables

    @property
    def evidential(self):
        """
        Returns the evidence as the attribute of the algorithm
        """
        return self._evidence

    @property
    def factors(self):
        return self._inner_model.factors

    @property
    def non_evidential(self):
        return tuple(var for var in self.variables if not var.is_evidential())

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

    def check_non_empty_query(self):
        if not self._query:
            raise AttributeError('query not specified')

    def check_one_variable_query(self):
        if len(self._query) > 1:
            raise ValueError('the query contains more than one variable')
        if len(self._query) < 1:
            raise ValueError('the query contains less than one variable')

    def check_query_and_evidence_intersection(self):
        if self._evidence:
            query_set = set(self._query)
            evidence_set = set(self._evidence)
            if not query_set.isdisjoint(evidence_set):
                raise ValueError(f'query variables {tuple(var.name for var in self._query)} and '
                                 f'evidential variables {tuple(var.name for var in self._evidence)} must be disjoint')

    def print_evidence(self):
        if self._evidence is not None:
            print('Evidence: ' + ', '.join(f'{var.name} = {var.domain[0]!r}' for var in self._evidence))
        else:
            print('No evidence')

    def print_pd(self):
        """
        Prints the complete probability distribution of the query variables
        """
        if self._distribution is not None:
            evidence_str = ' | ' + ', '.join(f'{var.name} = {var.domain[0]!r}' for var in self._evidence) \
                if self._evidence \
                else ''
            for values in Variable.evaluate_variables(self._query):
                query_str = 'P(' + ', '.join(f'{var.name} = {val!r}' for var, val in zip(self._query, values))
                value_str = str(self.pd(*values))
                equal_str = ') = '
                print(query_str + evidence_str + equal_str + value_str)
        else:
            raise AttributeError('distribution not computed')

    def print_query(self):
        if self._query is not None:
            print('Query: ' + ', '.join(variable.name for variable in self.query))
        else:
            print('No query')

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
        # Return the original domains of evidential variables and delete the evidence in factors
        self._delete_evidence()
        if evidence[0]:
            self._set_evidence(*evidence)
        else:
            self._evidence = ()
        self._set_evidence_tuples()
    
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

    def _clear_evidence(self):
        self._evidence = ()
        for inner_factor in self._inner_model.factors:
            inner_factor.clear_evidence()

    def _delete_evidence(self):
        for var in self._evidence:
            var.set_domain(self._inner_to_outer_variables[var].domain)
            for factor in var.factors:
                factor.delete_evidence(var)
        del self._evidence
        self._evidence = ()

    def _print_start(self):
        if self._print_info:
            print('*' * 40)
            print(f'{self._name} started')

    def _print_stop(self):
        if self._print_info:
            print(f'\n{self._name} stopped')
            print('*' * 40)

    def _set_evidence(self, *evidence_tuples):
        evidence_variables = tuple(var_val[0] for var_val in evidence_tuples)
        if len(evidence_variables) != len(set(evidence_variables)):
            raise ValueError(f'evidence must not contain duplicates')
        for outer_var, val in evidence_tuples:
            try:
                inner_var = self._outer_to_inner_variables[outer_var]
            except KeyError:
                # Clear the evidence also the evidence in the factors
                self._clear_evidence()
                raise ValueError(f'no model variable corresponds to evidential variable {outer_var.name}')
            try:
                inner_var.check_value(val)
            except ValueError as exception:
                # Clear the evidence also the evidence in the factors
                self._clear_evidence()
                raise exception
            # Set the new domain containing only one value
            inner_var.set_domain({val})
            # Add the evidence into its factors
            for inner_factor in inner_var.factors:
                inner_factor.add_evidence(inner_var)
        self._evidence = tuple(
            sorted(
                (self._outer_to_inner_variables[outer_var] for outer_var in evidence_variables),
                key=lambda x: x.name
            )
        )

    def _set_evidence_tuples(self):
        self._evidence_tuples = tuple((var, var.domain[0]) for var in self._evidence)

    def _set_query(self, *query_variables):
        # Check whether the query has duplicates
        if len(query_variables) != len(set(query_variables)):
            raise ValueError(f'query must not contain duplicates')
        for outer_var in query_variables:
            try:
                self._outer_to_inner_variables[outer_var]
            except KeyError:
                self._query = ()
                raise ValueError(f'no model variable corresponds to query variable {outer_var.name}')
        self._query = tuple(
            sorted(
                (self._outer_to_inner_variables[outer_var] for outer_var in query_variables),
                key=lambda x: x.name
            )
        )

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
