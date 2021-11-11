import pathlib
import sys

# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[3])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from pyb4ml.inference.factored.greedy_elimination import GBE
from pyb4ml.models import ExtendedStudent

eps = 1e-10

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
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g0, l0) =
# P(c) * P(i) * P(d | c) * P(g0 |d ,i) * P(l0 | g0) * P(j | l0, s) * P(s | i) * P(h | g0, j) =
# =>
# P(j, g0, l0) =
# P(l0 | g0) *
# (\sum_s P(j | l0, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g0, j) *
# [\sum_d P(g0 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g0 | d , i) * phi(d)):
# i0: 0.30 * 0.53 + 0.05 * 0.47 = 0.1825
# i1: 0.90 * 0.53 + 0.50 * 0.47 = 0.712
# =>
# (\sum_h P(h | g0, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.1825 + 0.3 * 0.20 * 0.712 = 0.1640825
# s1: 0.7 * 0.05 * 0.1825 + 0.3 * 0.80 * 0.712 = 0.1772675
# =>
# (\sum_s P(j | l0, s) * phi(s))
# j0: 0.95 * 0.1640825 + 0.25 * 0.1772675 = 0.20019525
# j1: 0.05 * 0.1640825 + 0.75 * 0.1772675 = 0.14115475
# =>
# P(l0 | g0) = 0.10
# =>
# P(j, g0, l0)
# j0: 0.4446675 * 0.10 = 0.04446675
# j1: 0.4498325 * 0.10 = 0.04498325
# =>
# P(j0 | g0, l0) = 0.20019525 / (0.20019525 + 0.14115475) = 0.5864808847224257
# P(j1 | g0, l0) = 0.14115475 / (0.20019525 + 0.14115475) = 0.41351911527757435
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.5864808847224257 / (1 + eps) <= pd('j0') <= 0.5864808847224257 * (1 + eps)
assert 0.41351911527757435 / (1 + eps) <= pd('j1') <= 0.41351911527757435 * (1 + eps)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g0'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g0, l1) =
# P(c) * P(i) * P(d | c) * P(g0 |d ,i) * P(l1 | g0) * P(j | l1, s) * P(s | i) * P(h | g0, j) =
# =>
# P(j, g0, l1) =
# P(l1 | g0) *
# (\sum_s P(j | l1, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g0, j) *
# [\sum_d P(g0 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g0 | d , i) * phi(d)):
# i0: 0.30 * 0.53 + 0.05 * 0.47 = 0.1825
# i1: 0.90 * 0.53 + 0.50 * 0.47 = 0.712
# =>
# (\sum_h P(h | g0, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.1825 + 0.3 * 0.20 * 0.712 = 0.1640825
# s1: 0.7 * 0.05 * 0.1825 + 0.3 * 0.80 * 0.712 = 0.1772675
# =>
# (\sum_s P(j | l1, s) * phi(s))
# j0: 0.65 * 0.1640825 + 0.15 * 0.1772675 = 0.13324375
# j1: 0.35 * 0.1640825 + 0.85 * 0.1772675 = 0.20810625
# =>
# P(l1 | g0) = 0.90
# =>
# P(j, g0, l1)
# j0: 0.13324375 * 0.90 = 0.119919375
# j1: 0.20810625 * 0.90 = 0.187295625
# =>
# P(j0 | g0, l1) = 0.119919375 / (0.119919375 + 0.187295625) = 0.39034348908744687
# P(j1 | g0, l1) = 0.187295625 / (0.119919375 + 0.187295625) = 0.609656510912553
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.39034348908744687 / (1 + eps) <= pd('j0') <= 0.39034348908744687 * (1 + eps)
assert 0.609656510912553 / (1 + eps) <= pd('j1') <= 0.609656510912553 * (1 + eps)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g1'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g1, l0) =
# P(c) * P(i) * P(d | c) * P(g1 |d ,i) * P(l0 | g1) * P(j | l0, s) * P(s | i) * P(h | g1, j) =
# =>
# P(j, g1, l0) =
# P(l0 | g1) *
# (\sum_s P(j | l0, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g1, j) *
# [\sum_d P(g1 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g1 | d , i) * phi(d)):
# i0: 0.40 * 0.53 + 0.25 * 0.47 = 0.3295
# i1: 0.08 * 0.53 + 0.30 * 0.47 = 0.1834
# =>
# (\sum_h P(h | g1, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.3295 + 0.3 * 0.20 * 0.1834 = 0.2301215
# s1: 0.7 * 0.05 * 0.3295 + 0.3 * 0.80 * 0.1834 = 0.0555485
# =>
# (\sum_s P(j | l0, s) * phi(s))
# j0: 0.95 * 0.2301215 + 0.25 * 0.0555485 = 0.23250255
# j1: 0.05 * 0.2301215 + 0.75 * 0.0555485 = 0.05316745
# =>
# P(l0 | g1) = 0.40
# =>
# P(j, g1, l0)
# j0: 0.23250255 * 0.40 = 0.09300102
# j1: 0.05316745 * 0.40 = 0.02126698
# =>
# P(j0 | g1, l0) = 0.09300102 / (0.09300102 + 0.02126698) = 0.8138850771869639
# P(j1 | g1, l0) = 0.02126698 / (0.09300102 + 0.02126698) = 0.18611492281303602
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.813885077186964 / (1 + eps) <= pd('j0') <= 0.813885077186964 * (1 + eps)
assert 0.18611492281303602 / (1 + eps) <= pd('j1') <= 0.18611492281303602 * (1 + eps)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g1'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g1, l1) =
# P(c) * P(i) * P(d | c) * P(g1 |d ,i) * P(l0 | g1) * P(j | l1, s) * P(s | i) * P(h | g1, j) =
# =>
# P(j, g1, l1) =
# P(l1 | g1) *
# (\sum_s P(j | l1, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g1, j) *
# [\sum_d P(g1 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g1 | d , i) * phi(d)):
# i0: 0.40 * 0.53 + 0.25 * 0.47 = 0.3295
# i1: 0.08 * 0.53 + 0.30 * 0.47 = 0.1834
# =>
# (\sum_h P(h | g1, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.3295 + 0.3 * 0.20 * 0.1834 = 0.2301215
# s1: 0.7 * 0.05 * 0.3295 + 0.3 * 0.80 * 0.1834 = 0.0555485
# =>
# (\sum_s P(j | l1, s) * phi(s))
# j0: 0.65 * 0.2301215 + 0.15 * 0.0555485 = 0.15791125
# j1: 0.35 * 0.2301215 + 0.85 * 0.0555485 = 0.12775875
# =>
# P(l1 | g1) = 0.60
# =>
# P(j, g1, l1)
# j0: 0.15791125 * 0.60 = 0.09474675
# j1: 0.12775875 * 0.60 = 0.07665525
# =>
# P(j0 | g1, l1) = 0.09474675 / (0.09474675 + 0.07665525) = 0.5527750551335457
# P(j1 | g1, l1) = 0.07665525 / (0.09474675 + 0.07665525) = 0.44722494486645425
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.5527750551335457 / (1 + eps) <= pd('j0') <= 0.5527750551335457 * (1 + eps)
assert 0.44722494486645425 / (1 + eps) <= pd('j1') <= 0.44722494486645425 * (1 + eps)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g2'), (letter, 'l0'))
algorithm.run(cost='weighted-min-fill')
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g2, l0) =
# P(c) * P(i) * P(d | c) * P(g2 |d ,i) * P(l0 | g2) * P(j | l0, s) * P(s | i) * P(h | g2, j) =
# =>
# P(j, g2, l0) =
# P(l0 | g2) *
# (\sum_s P(j | l0, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g2, j) *
# [\sum_d P(g2 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g2 | d , i) * phi(d)):
# i0: 0.30 * 0.53 + 0.7 * 0.47 = 0.488
# i1: 0.02 * 0.53 + 0.20 * 0.47 = 0.1046
# =>
# (\sum_h P(h | g2, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.488 + 0.3 * 0.20 * 0.1046 = 0.330796
# s1: 0.7 * 0.05 * 0.488 + 0.3 * 0.80 * 0.1046 = 0.042184
# =>
# (\sum_s P(j | l0, s) * phi(s))
# j0: 0.95 * 0.330796 + 0.25 * 0.042184 = 0.3248022
# j1: 0.05 * 0.330796 + 0.75 * 0.042184 = 0.0481778
# =>
# P(l0 | g2) = 0.99
# =>
# P(j, g2, l0)
# j0: 0.3248022 * 0.99 = 0.321554178
# j1: 0.0481778 * 0.99 = 0.047696022
# =>
# P(j0 | g2, l0) = 0.321554178 / (0.321554178 + 0.047696022) = 0.8708300713174969
# P(j1 | g2, l0) = 0.047696022 / (0.321554178 + 0.047696022) = 0.12916992868250307
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.8708300713174969 / (1 + eps) <= pd('j0') <= 0.8708300713174969 * (1 + eps)
assert 0.12916992868250307 / (1 + eps) <= pd('j1') <= 0.12916992868250307 * (1 + eps)

algorithm.set_query(job)
algorithm.set_evidence((grade, 'g2'), (letter, 'l1'))
algorithm.run(cost='weighted-min-fill', print_info=True)
algorithm.print_ordering()
algorithm.print_pd()
# P(c, d, h, i, s, j, g2, l1) =
# P(c) * P(i) * P(d | c) * P(g2 |d ,i) * P(l1 | g2) * P(j | l1, s) * P(s | i) * P(h | g2, j) =
# =>
# P(j, g2, l1) =
# P(l1 | g2) *
# (\sum_s P(j | l1, s) *
# [\sum_i P(s | i) *
# (\sum_h P(h | g2, j) *
# [\sum_d P(g2 | d , i) *
# (\sum_c P(c) * P(d | c))])])
# =>
# (\sum_c P(c) * P(d | c)):
# d0: 0.2 * 0.2 + 0.5 * 0.5 + 0.3 * 0.8 = 0.53
# d1: 0.2 * 0.8 + 0.5 * 0.5 + 0.3 * 0.2 = 0.47
# =>
# (\sum_d P(g2 | d , i) * phi(d)):
# i0: 0.30 * 0.53 + 0.7 * 0.47 = 0.488
# i1: 0.02 * 0.53 + 0.20 * 0.47 = 0.1046
# =>
# (\sum_h P(h | g2, j)) = 1
# =>
# (\sum_i P(i) * P(s | i) * phi(i))
# s0: 0.7 * 0.95 * 0.488 + 0.3 * 0.20 * 0.1046 = 0.330796
# s1: 0.7 * 0.05 * 0.488 + 0.3 * 0.80 * 0.1046 = 0.042184
# =>
# (\sum_s P(j | l1, s) * phi(s))
# j0: 0.65 * 0.330796 + 0.15 * 0.042184 = 0.221345
# j1: 0.35 * 0.330796 + 0.85 * 0.042184 = 0.151635
# =>
# P(l1 | g2) = 0.01
# =>
# P(j, g2, l1)
# j0: 0.221345 * 0.01 = 0.00221345
# j1: 0.151635 * 0.01 = 0.00151635
# =>
# P(j0 | g2, l1) = 0.00221345 / (0.00221345 + 0.00151635) = 0.5934500509410692
# P(j1 | g2, l1) = 0.00151635 / (0.00221345 + 0.00151635) = 0.40654994905893077
assert algorithm.ordering == (coherence, difficulty, happy, intelligence, sat)
pd = algorithm.pd
assert 0.5934500509410692 / (1 + eps) <= pd('j0') <= 0.5934500509410692 * (1 + eps)
assert 0.40654994905893077 / (1 + eps) <= pd('j1') <= 0.40654994905893077 * (1 + eps)