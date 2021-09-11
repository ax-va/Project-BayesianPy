from pyb4ml.algorithms.inference.sum_product import SumProduct
from pyb4ml.models.factor_graphs.student import Student

model = Student()
algorithm = SumProduct(factorization=model.factorization)

for query in model.factorization.factors:
    print('query:', query)
    algorithm.run(print_passing=True)
    for value in query.domain:
        print(f'P({query}={value!r})={algorithm.pd(value)}')