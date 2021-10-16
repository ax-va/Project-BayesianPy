from pyb4ml.inference.factored.greedy_ordering import GO
from pyb4ml.models.factor_graphs.extended_student import ExtendedStudent

model = ExtendedStudent()
coherence = model.get_variable('Coherence')
difficulty = model.get_variable('Difficulty')
happy = model.get_variable('Happy')
intelligence = model.get_variable('Intelligence')
grade = model.get_variable('Grade')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
job = model.get_variable('Job')

algorithm = GO(model)
algorithm.set_query(job)
algorithm.run(cost='weighted-min-fill')
algorithm.print_ordering()
assert algorithm.ordering == [coherence, difficulty, happy, intelligence, grade, letter, sat]