from pyb4ml.algorithms import BEA
from pyb4ml.models import Student

# Test the Bucket Elimination Algorithm on the Student model
model = Student()
algorithm = BEA(model)  # Bucket Elimination Algorithm

difficulty = model.get_variable('Difficulty')
intelligence = model.get_variable('Intelligence')
grade = model.get_variable('Grade')
sat = model.get_variable('SAT')
letter = model.get_variable('Letter')

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.set_elimination_order([letter, sat, intelligence, grade])

algorithm.run()
pd = algorithm.pd

print(pd('d0'))
print(pd('d1'))
