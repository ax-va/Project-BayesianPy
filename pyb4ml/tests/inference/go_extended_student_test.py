import pathlib
import sys

# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[3])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from pyb4ml.inference.factored.greedy_ordering import GO
from pyb4ml.models import ExtendedStudent

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

algorithm.set_query(coherence)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (happy, job, letter, sat, grade, intelligence, difficulty)

algorithm.set_query(difficulty)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, happy, job, letter, sat, grade, intelligence)

algorithm.set_query(grade)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, job, letter, sat)

algorithm.set_query(happy)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, intelligence, letter, sat, grade, job)

algorithm.set_query(intelligence)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, job, letter, grade, sat)

algorithm.set_query(job)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, grade, letter, sat)

algorithm.set_query(letter)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, grade, job, sat)

algorithm.set_query(sat)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, grade, job, letter)

algorithm.set_query(happy, job)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, intelligence, letter, sat, grade)

algorithm.set_query(difficulty, intelligence)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, happy, job, letter, sat, grade)

algorithm.set_query(grade, sat)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, job, letter)

algorithm.set_query(difficulty, happy)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, letter, intelligence, sat, grade, job)

algorithm.set_query(coherence, happy)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (letter, intelligence, sat, grade, job, difficulty)

algorithm.set_query(grade, happy)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, intelligence, letter, sat, job)

algorithm.set_query(coherence, grade)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (happy, job, letter, sat, intelligence, difficulty)

algorithm.set_query(intelligence, letter)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, job, grade, sat)

algorithm.set_query(happy, letter)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, intelligence, sat, grade, job)

algorithm.set_query(coherence, difficulty, happy)
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (letter, intelligence, sat, grade, job)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, letter, sat)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, sat)

algorithm.set_query(job)
algorithm.set_evidence(None)
algorithm.run(cost='min-fill')
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, grade, letter, sat)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'))
algorithm.run(cost='min-fill', print_info=True)
algorithm.print_order()
assert algorithm.order == (coherence, difficulty, happy, intelligence, letter, sat)
