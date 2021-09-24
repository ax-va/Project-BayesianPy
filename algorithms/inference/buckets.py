class Bucket:
    def __init__(self):
        self._content = []

    @property
    def content(self):
        return self._content

    def add(self, factor):
        self._content.append(factor)


class Buckets:
    def __init__(self, be_algorithm):
        self._buckets = {}
        self._factor_cache = be_algorithm.factor_cache
        self._fill_buckets(be_algorithm.elimination_order)
        self._fill_buckets(be_algorithm.query)        
                          
    def _fill_buckets(self, variables):
        for variable in variables:        
            self._buckets[variable] = Bucket()
            # Fill the bucket with factors and delete the factors from the factorization cache
            for factor_variables in self._factor_cache.keys():
                if variable in factor_variables:
                    # Add the factor into the bucket
                    self._buckets[variable].add(self._factor_cache[factor_variables])
                    # Reduce the factorization cache by the factor
                    del self._factor_cache[factor_variables]
