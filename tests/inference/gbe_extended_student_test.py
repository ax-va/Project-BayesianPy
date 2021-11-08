import pathlib
import sys

# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[3])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)


from pyb4ml.inference.factored.greedy_elimination import GBE
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
algorithm = GBE(model)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g1'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g1'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g2'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g2'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_pd()