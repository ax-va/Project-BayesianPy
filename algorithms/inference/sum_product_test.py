from pyb4ml.algorithms.inference.sum_product import SumProduct
from pyb4ml.models.factor_graphs.student import Student

# Test on the Student model
model = Student()
algorithm = SumProduct(factorization=model.factorization)

eps = 1 / 1e10

for query in model.variables:
    print('query:', query)
    algorithm.set_query(query)
    algorithm.run(print_messages=True, print_loop_passing=True)
    print('-' * 20)
    print('probability distribution:')
    for value in query.domain:
        print(f'P({query}={value!r})={algorithm.pd(value)}')
    print('-'*20)
    print('-'*20)