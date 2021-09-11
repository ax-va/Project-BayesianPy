from pyb4ml.algorithms.inference.sum_product import SumProduct
from pyb4ml.models.factor_graphs.student import Student

model = Student()
algorithm = SumProduct(factorization=model.factorization)

for query in model.factorization.variables:
    print('query:', query)
    algorithm.set_query(query)
    algorithm.run(print_messages=True, print_loop_passing=True)
    for value in query.domain:
        print(f'P({query}={value!r})={algorithm.pd(value)}')