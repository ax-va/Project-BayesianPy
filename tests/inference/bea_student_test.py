from pyb4ml.algorithms import BEA, BPA
from pyb4ml.models import Student

# Test the Bucket Elimination Algorithm on the Student model
model = Student()
algorithm = BEA(model)  # Bucket Elimination Algorithm

eps = 1 / 1e12

difficulty = model.get_variable('Difficulty')
intelligence = model.get_variable('Intelligence')
grade = model.get_variable('Grade')
sat = model.get_variable('SAT')
letter = model.get_variable('Letter')

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run(print_info=True)
pd = algorithm.pd
assert 0.3972483414607588 - eps < pd('d0') < 0.3972483414607588 + eps
assert 0.6027516585392411 - eps < pd('d1') < 0.6027516585392411 + eps


# algorithm = BPA(model)  # Bucket Elimination Algorithm
# algorithm.set_query(difficulty)
# algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
# algorithm.run()
# pd = algorithm.pd
#
# print(pd('d0'))
# print(pd('d1'))
