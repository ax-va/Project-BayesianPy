from pyb4ml.algorithms.inference.sum_product import SumProduct
from pyb4ml.models.factor_graphs.student import Student

student_model = Student()
algorithm = SumProduct(factorization=student_model.factorization)

for query in student_model.factorization.factors:
    print('query:', query)
    algorithm.run()
    for value in query.domain:
        print(f'P({query}={value!r})={algorithm.pd(value)}')