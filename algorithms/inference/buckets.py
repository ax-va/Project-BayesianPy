class Bucket:
    def __init__(self):
        self._content = []

    @property
    def content(self):
        return self._content

    def add(self, factor):
        self._content.append(factor)


class Buckets:
    def __init__(self, elimination_order, factorization):
        self._buckets = {}
        for variable in elimination_order:
            self._buckets[variable] = Bucket()
            # Fill the bucket with factors and delete the factors from the factorization
            for factor_variables in factorization.cache.keys():
                if variable in factor_variables:
                    # Add the factor into the bucket
                    self._buckets[variable].add(factorization.cache[factor_variables])
                    # Reduce the factorization cache by the factor
                    del factorization.cache[factor_variables]
