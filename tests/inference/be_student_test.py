from pyb4ml.inference import BE
from pyb4ml.models import Student

# Test the Bucket Elimination algorithm on the Student model
model = Student()
difficulty = model.get_variable('Difficulty')
intelligence = model.get_variable('Intelligence')
grade = model.get_variable('Grade')
sat = model.get_variable('SAT')
letter = model.get_variable('Letter')

algorithm = BE(model)  # Bucket Elimination algorithm

eps = 1e-12

# Test conditional distributions

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's0'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.4742196406430358 / (1 + eps) <= pd('d0') <= 0.4742196406430358 * (1 + eps)
assert 0.5257803593569642 / (1 + eps) <= pd('d1') <= 0.5257803593569642 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.3972483414607588 / (1 + eps) <= pd('d0') <= 0.3972483414607588 * (1 + eps)
assert 0.6027516585392411 / (1 + eps) <= pd('d1') <= 0.6027516585392411 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's0'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.7737141941302315 / (1 + eps) <= pd('d0') <= 0.7737141941302315 * (1 + eps)
assert 0.22628580586976843 / (1 + eps) <= pd('d1') <= 0.22628580586976843 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's1'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.6790559493929356 / (1 + eps) <= pd('d0') <= 0.6790559493929356 * (1 + eps)
assert 0.32094405060706443 / (1 + eps) <= pd('d1') <= 0.32094405060706443 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.4622878086419753 / (1 + eps) <= pd('d0') <= 0.4622878086419753 * (1 + eps)
assert 0.5377121913580247 / (1 + eps) <= pd('d1') <= 0.5377121913580247 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.7364313925340807 / (1 + eps) <= pd('d0') <= 0.7364313925340807 * (1 + eps)
assert 0.26356860746591926 / (1 + eps) <= pd('d1') <= 0.26356860746591926 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's0'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The trail Difficulty - Grade - Intelligence - SAT
# is not active given the empty set of observed variables
# because neither Grade nor Letter is observed
assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)

algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's1'))
algorithm.set_ordering([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The trail Difficulty - Grade - Intelligence - SAT
# is not active given the empty set of observed variables
# because neither Grade nor Letter is observed
assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l0'), (sat, 's0'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.07627202787313875 / (1 + eps) <= pd('g0') <= 0.07627202787313875 * (1 + eps)
assert 0.32590872028474893 / (1 + eps) <= pd('g1') <= 0.32590872028474893 * (1 + eps)
assert 0.5978192518421124 / (1 + eps) <= pd('g2') <= 0.5978192518421124 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l0'), (sat, 's1'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.4434481273371577 / (1 + eps) <= pd('g0') <= 0.4434481273371577 * (1 + eps)
assert 0.2599996084343245 / (1 + eps) <= pd('g1') <= 0.2599996084343245 * (1 + eps)
assert 0.29655226422851777 / (1 + eps) <= pd('g2') <= 0.29655226422851777 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l1'), (sat, 's0'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.5810710656407828 / (1 + eps) <= pd('g0') <= 0.5810710656407828 * (1 + eps)
assert 0.4138173427364206 / (1 + eps) <= pd('g1') <= 0.4138173427364206 * (1 + eps)
assert 0.005111591622796632 / (1 + eps) <= pd('g2') <= 0.005111591622796632 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l1'), (sat, 's1'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.9103575782746748 / (1 + eps) <= pd('g0') <= 0.9103575782746748 * (1 + eps)
assert 0.08895915113677469 / (1 + eps) <= pd('g1') <= 0.08895915113677469 * (1 + eps)
assert 0.0006832705885505288 / (1 + eps) <= pd('g2') <= 0.0006832705885505288 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l0'), (sat, 's0'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.011442786069651743 / (1 + eps) <= pd('g0') <= 0.011442786069651743 * (1 + eps)
assert 0.13333333333333333 / (1 + eps) <= pd('g1') <= 0.13333333333333333 * (1 + eps)
assert 0.8552238805970149 / (1 + eps) <= pd('g2') <= 0.8552238805970149 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l0'), (sat, 's1'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.10473118279569894 / (1 + eps) <= pd('g0') <= 0.10473118279569894 * (1 + eps)
assert 0.2778494623655915 / (1 + eps) <= pd('g1') <= 0.2778494623655915 * (1 + eps)
assert 0.6174193548387096 / (1 + eps) <= pd('g2') <= 0.6174193548387096 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l1'), (sat, 's0'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.33047895500725694 / (1 + eps) <= pd('g0') <= 0.33047895500725694 * (1 + eps)
assert 0.641799709724238 / (1 + eps) <= pd('g1') <= 0.641799709724238 * (1 + eps)
assert 0.027721335268505072 / (1 + eps) <= pd('g2') <= 0.027721335268505072 * (1 + eps)

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l1'), (sat, 's1'))
algorithm.set_ordering([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# The values for the assertions are obtained using the BPA
assert 0.690236220472441 / (1 + eps) <= pd('g0') <= 0.690236220472441 * (1 + eps)
assert 0.30519685039370076 / (1 + eps) <= pd('g1') <= 0.30519685039370076 * (1 + eps)
assert 0.004566929133858268 / (1 + eps) <= pd('g2') <= 0.004566929133858268 * (1 + eps)

# Test marginal joint distributions

algorithm.set_query(difficulty, intelligence)
algorithm.set_evidence(None)
algorithm.set_ordering([letter, sat, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(d,i) = P(d) * P(i)
# =>
# P(d0,i0) = P(d0) * P(i0)
# = 0.6 * 0.7 = 0.42
# P(d0,i1) = P(d0) * P(i1)
# = 0.6 * 0.3 = 0.18
# P(d1,i0) = P(d1) * P(i0)
# = 0.4 * 0.7 = 0.28
# P(d1,i1) = P(d1) * P(i1)
# = 0.4 * 0.3 = 0.12
assert 0.42 / (1 + eps) <= pd('d0', 'i0') <= 0.42 * (1 + eps)
assert 0.18 / (1 + eps) <= pd('d0', 'i1') <= 0.18 * (1 + eps)
assert 0.28 / (1 + eps) <= pd('d1', 'i0') <= 0.28 * (1 + eps)
assert 0.12 / (1 + eps) <= pd('d1', 'i1') <= 0.12 * (1 + eps)

# Test conditional joint distributions

algorithm.set_query(difficulty, intelligence)
algorithm.set_evidence((letter, 'l0'), (sat, 's0'))
algorithm.set_ordering([letter, sat, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(d,i|s0,l0) = P(d,i,s0,l0) / P(s0,l0)
# P(d,i,s0,l0) = \sum_{g} P(d,i,g,s0,l0)
# = P(d) * P(i) * P(s0|i) * \sum_{g} P(g|d,i) * P(l0|g)
# =>
# P(d0,i0,s0,l0)
# = P(d0) * P(i0) * P(s0|i0)
# * (P(g0|d0,i0) * P(l0|g0) + P(g1|d0,i0) * P(l0|g1) + P(g2|d0,i0) * P(l0|g2))
# = 0.6 * 0.7 * 0.95
# * (0.3 * 0.1 + 0.4 * 0.4 + 0.3 * 0.99)
# = 0.194313
#
# P(d0,i1,s0,l0)
# = P(d0) * P(i1) * P(s0|i1)
# * (P(g0|d0,i1) * P(l0|g0) + P(g1|d0,i1) * P(l0|g1) + P(g2|d0,i1) * P(l0|g2))
# = 0.6 * 0.3 * 0.2
# * (0.9 * 0.1 + 0.08 * 0.4 + 0.02 * 0.99)
# = 0.0051048
#
# P(d1,i0,s0,l0)
# = P(d1) * P(i0) * P(s0|i0)
# * (P(g0|d1,i0) * P(l0|g0) + P(g1|d1,i0) * P(l0|g1) + P(g2|d1,i0) * P(l0|g2))
# = 0.4 * 0.7 * 0.95
# * (0.05 * 0.1 + 0.25 * 0.4 + 0.7 * 0.99)
# = 0.212268
#
# P(d1,i1,s0,l0)
# = P(d1) * P(i1) * P(s0|i1)
# * (P(g0|d1,i1) * P(l0|g0) + P(g1|d1,i1) * P(l0|g1) + P(g2|d1,i1) * P(l0|g2))
# = 0.4 * 0.3 * 0.2
# * (0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.99)
# = 0.008832
# =>
# P(s0,l0) = P(d0,i0,s0,l0) + P(d0,i1,s0,l0) + P(d1,i0,s0,l0) + P(d1,i1,s0,l0)
# = 0.194313 + 0.0051048 + 0.212268 + 0.008832 = 0.4205178
# =>
# P(d0,i0|s0,l0) = P(d0,i0,s0,l0) / P(s0,l0)
# = 0.194313 / 0.4205178 = 0.4620803209757114
#
# P(d0,i1|s0,l0) = P(d0,i1,s0,l0) / P(s0,l0)
# = 0.0051048 / 0.4205178 = 0.012139319667324426
#
# P(d1,i0|s0,l0) = P(d1,i0,s0,l0) / P(s0,l0)
# = 0.212268 / 0.4205178 = 0.5047776812301406
#
# P(d1,i1|s0,l0) = P(d1,i1,s0,l0) / P(s0,l0)
# = 0.008832 / 0.4205178 = 0.021002678126823642
assert 0.4620803209757114 / (1 + eps) <= pd('d0', 'i0') <= 0.4620803209757114 * (1 + eps)
assert 0.012139319667324426 / (1 + eps) <= pd('d0', 'i1') <= 0.012139319667324426 * (1 + eps)
assert 0.5047776812301406 / (1 + eps) <= pd('d1', 'i0') <= 0.5047776812301406 * (1 + eps)
assert 0.021002678126823642 / (1 + eps) <= pd('d1', 'i1') <= 0.021002678126823642 * (1 + eps)

algorithm.set_query(difficulty, intelligence)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.set_ordering([letter, sat, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(d,i|s1,l0) = P(d,i,s1,l0) / P(s1,l0)
# P(d,i,s1,l0) = \sum_{g} P(d,i,g,s1,l0)
# = P(d) * P(i) * P(s1|i) * \sum_{g} P(g|d,i) * P(l0|g)
# =>
# P(d0,i0,s1,l0)
# = P(d0) * P(i0) * P(s1|i0)
# * (P(g0|d0,i0) * P(l0|g0) + P(g1|d0,i0) * P(l0|g1) + P(g2|d0,i0) * P(l0|g2))
# = 0.6 * 0.7 * 0.05
# * (0.3 * 0.1 + 0.4 * 0.4 + 0.3 * 0.99)
# = 0.010227
#
# P(d0,i1,s1,l0)
# = P(d0) * P(i1) * P(s1|i1)
# * (P(g0|d0,i1) * P(l0|g0) + P(g1|d0,i1) * P(l0|g1) + P(g2|d0,i1) * P(l0|g2))
# = 0.6 * 0.3 * 0.8
# * (0.9 * 0.1 + 0.08 * 0.4 + 0.02 * 0.99)
# = 0.0204192
#
# P(d1,i0,s1,l0)
# = P(d1) * P(i0) * P(s1|i0)
# * (P(g0|d1,i0) * P(l0|g0) + P(g1|d1,i0) * P(l0|g1) + P(g2|d1,i0) * P(l0|g2))
# = 0.4 * 0.7 * 0.05
# * (0.05 * 0.1 + 0.25 * 0.4 + 0.7 * 0.99)
# = 0.011172
#
# P(d1,i1,s1,l0)
# = P(d1) * P(i1) * P(s1|i1)
# * (P(g0|d1,i1) * P(l0|g0) + P(g1|d1,i1) * P(l0|g1) + P(g2|d1,i1) * P(l0|g2))
# = 0.4 * 0.3 * 0.8
# * (0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.99)
# = 0.035328
# =>
# P(s0,l0) = P(d0,i0,s1,l0) + P(d0,i1,s1,l0) + P(d1,i0,s1,l0) + P(d1,i1,s1,l0)
# = 0.010227 + 0.0204192 + 0.011172 + 0.035328 = 0.0771462
# =>
# P(d0,i0|s1,l0) = P(d0,i0,s1,l0) / P(s1,l0)
# = 0.010227 / 0.0771462 = 0.13256647767485633
#
# P(d0,i1|s1,l0) = P(d0,i1,s1,l0) / P(s1,l0)
# = 0.0204192 / 0.0771462 = 0.2646818637859026
#
# P(d1,i0|s1,l0) = P(d1,i0,s1,l0) / P(s1,l0)
# = 0.011172 / 0.0771462 = 0.14481594686452476
#
# P(d1,i1|s1,l0) = P(d1,i1,s1,l0) / P(s1,l0)
# = 0.035328 / 0.0771462 = 0.4579357116747163
assert 0.13256647767485633 / (1 + eps) <= pd('d0', 'i0') <= 0.13256647767485633 * (1 + eps)
assert 0.2646818637859026 / (1 + eps) <= pd('d0', 'i1') <= 0.2646818637859026 * (1 + eps)
assert 0.14481594686452476 / (1 + eps) <= pd('d1', 'i0') <= 0.14481594686452476 * (1 + eps)
assert 0.4579357116747163 / (1 + eps) <= pd('d1', 'i1') <= 0.4579357116747163 * (1 + eps)

algorithm.set_query(difficulty, intelligence)
algorithm.set_evidence((letter, 'l1'), (sat, 's0'))
algorithm.set_ordering([letter, sat, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(d,i|s0,l1) = P(d,i,s0,l1) / P(s0,l1)
# P(d,i,s0,l1) = \sum_{g} P(d,i,g,s0,l1)
# = P(d) * P(i) * P(s0|i) * \sum_{g} P(g|d,i) * P(l1|g)
# =>
# P(d0,i0,s0,l1)
# = P(d0) * P(i0) * P(s0|i0)
# * (P(g0|d0,i0) * P(l1|g0) + P(g1|d0,i0) * P(l1|g1) + P(g2|d0,i0) * P(l1|g2))
# = 0.6 * 0.7 * 0.95
# * (0.3 * 0.9 + 0.4 * 0.6 + 0.3 * 0.01)
# = 0.204687
#
# P(d0,i1,s0,l1)
# = P(d0) * P(i1) * P(s0|i1)
# * (P(g0|d0,i1) * P(l1|g0) + P(g1|d0,i1) * P(l1|g1) + P(g2|d0,i1) * P(l1|g2))
# = 0.6 * 0.3 * 0.2
# * (0.9 * 0.9 + 0.08 * 0.6 + 0.02 * 0.01)
# = 0.0308952
#
# P(d1,i0,s0,l1)
# = P(d1) * P(i0) * P(s0|i0)
# * (P(g0|d1,i0) * P(l1|g0) + P(g1|d1,i0) * P(l1|g1) + P(g2|d1,i0) * P(l1|g2))
# = 0.4 * 0.7 * 0.95
# * (0.05 * 0.9 + 0.25 * 0.6 + 0.7 * 0.01)
# = 0.053732
#
# P(d1,i1,s0,l1)
# = P(d1) * P(i1) * P(s0|i1)
# * (P(g0|d1,i1) * P(l1|g0) + P(g1|d1,i1) * P(l1|g1) + P(g2|d1,i1) * P(l1|g2))
# = 0.4 * 0.3 * 0.2
# * (0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.01)
# = 0.015168
# =>
# P(s0,l1) = P(d0,i0,s0,l1) + P(d0,i1,s0,l1) + P(d1,i0,s0,l1) + P(d1,i1,s0,l1)
# = 0.204687 + 0.0308952 + 0.053732 + 0.015168 = 0.3044822
# =>
# P(d0,i0|s0,l1) = P(d0,i0,s0,l1) / P(s0,l1)
# = 0.204687 / 0.3044822 = 0.6722461937019636
#
# P(d0,i1|s0,l1) = P(d0,i1,s0,l1) / P(s0,l1)
# = 0.0308952 / 0.3044822 = 0.10146800042826806
#
# P(d1,i0|s0,l1) = P(d1,i0,s0,l1) / P(s0,l1)
# = 0.053732 / 0.3044822 = 0.176470085936058
#
# P(d1,i1|s0,l1) = P(d1,i1,s0,l1) / P(s0,l1)
# = 0.015168 / 0.3044822 = 0.04981571993371041
assert 0.6722461937019636 / (1 + eps) <= pd('d0', 'i0') <= 0.6722461937019636 * (1 + eps)
assert 0.10146800042826806 / (1 + eps) <= pd('d0', 'i1') <= 0.10146800042826806 * (1 + eps)
assert 0.176470085936058 / (1 + eps) <= pd('d1', 'i0') <= 0.176470085936058 * (1 + eps)
assert 0.04981571993371041 / (1 + eps) <= pd('d1', 'i1') <= 0.04981571993371041 * (1 + eps)

algorithm.set_query(difficulty, intelligence)
algorithm.set_evidence((letter, 'l1'), (sat, 's1'))
algorithm.set_ordering([letter, sat, grade])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(d,i|s1,l1) = P(d,i,s1,l1) / P(s1,l1)
# P(d,i,s1,l1) = \sum_{g} P(d,i,g,s1,l1)
# = P(d) * P(i) * P(s1|i) * \sum_{g} P(g|d,i) * P(l1|g)
# =>
# P(d0,i0,s1,l1)
# = P(d0) * P(i0) * P(s1|i0)
# * (P(g0|d0,i0) * P(l1|g0) + P(g1|d0,i0) * P(l1|g1) + P(g2|d0,i0) * P(l1|g2))
# = 0.6 * 0.7 * 0.05
# * (0.3 * 0.9 + 0.4 * 0.6 + 0.3 * 0.01)
# = 0.010773
#
# P(d0,i1,s1,l1)
# = P(d0) * P(i1) * P(s1|i1)
# * (P(g0|d0,i1) * P(l1|g0) + P(g1|d0,i1) * P(l1|g1) + P(g2|d0,i1) * P(l1|g2))
# = 0.6 * 0.3 * 0.8
# * (0.9 * 0.9 + 0.08 * 0.6 + 0.02 * 0.01)
# = 0.1235808
#
# P(d1,i0,s1,l1)
# = P(d1) * P(i0) * P(s1|i0)
# * (P(g0|d1,i0) * P(l1|g0) + P(g1|d1,i0) * P(l1|g1) + P(g2|d1,i0) * P(l1|g2))
# = 0.4 * 0.7 * 0.05
# * (0.05 * 0.9 + 0.25 * 0.6 + 0.7 * 0.01)
# = 0.002828
#
# P(d1,i1,s1,l1)
# = P(d1) * P(i1) * P(s1|i1)
# * (P(g0|d1,i1) * P(l1|g0) + P(g1|d1,i1) * P(l1|g1) + P(g2|d1,i1) * P(l1|g2))
# = 0.4 * 0.3 * 0.8
# * (0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.01)
# = 0.060672
# =>
# P(s1,l1) = P(d0,i0,s1,l1) + P(d0,i1,s1,l1) + P(d1,i0,s1,l1) + P(d1,i1,s1,l1)
# = 0.010773 + 0.1235808 + 0.002828 + 0.060672 = 0.1978538
# =>
# P(d0,i0|s1,l1) = P(d0,i0,s1,l1) / P(s1,l1)
# = 0.010773 / 0.1978538 = 0.0544492953888174
#
# P(d0,i1|s1,l1) = P(d0,i1,s1,l1) / P(s1,l1)
# = 0.1235808 / 0.1978538 = 0.6246066540041182
#
# P(d1,i0|s1,l1) = P(d1,i0,s1,l1) / P(s1,l1)
# = 0.002828 / 0.1978538 = 0.014293382285303592
#
# P(d1,i1|s1,l1) = P(d1,i1,s1,l1) / P(s1,l1)
# = 0.060672 / 0.1978538 = 0.3066506683217608
assert 0.0544492953888174 / (1 + eps) <= pd('d0', 'i0') <= 0.0544492953888174 * (1 + eps)
assert 0.6246066540041182 / (1 + eps) <= pd('d0', 'i1') <= 0.6246066540041182 * (1 + eps)
assert 0.014293382285303592 / (1 + eps) <= pd('d1', 'i0') <= 0.0142933822853035928 * (1 + eps)
assert 0.3066506683217608 / (1 + eps) <= pd('d1', 'i1') <= 0.3066506683217608 * (1 + eps)

# Test marginal joint probability distributions again
algorithm.set_query(letter, sat)
algorithm.set_evidence(None)
algorithm.set_ordering([difficulty, intelligence, grade])
algorithm.run(print_info=True)
pd = algorithm.pd
algorithm.print_pd()
# P(l,s) =
# \sum_{d,i,g} P(g|d,i) * P(d) * P(i) * P(s|i) * P(l|g)
# = P(g0|d0,i0) * P(d0) * P(i0) * P(s|i0) * P(l|g0)
# + P(g1|d0,i0) * P(d0) * P(i0) * P(s|i0) * P(l|g1)
# + P(g2|d0,i0) * P(d0) * P(i0) * P(s|i0) * P(l|g2)
# + P(g0|d0,i1) * P(d0) * P(i1) * P(s|i1) * P(l|g0)
# + P(g1|d0,i1) * P(d0) * P(i1) * P(s|i1) * P(l|g1)
# + P(g2|d0,i1) * P(d0) * P(i1) * P(s|i1) * P(l|g2)
# + P(g0|d1,i0) * P(d1) * P(i0) * P(s|i0) * P(l|g0)
# + P(g1|d1,i0) * P(d1) * P(i0) * P(s|i0) * P(l|g1)
# + P(g2|d1,i0) * P(d1) * P(i0) * P(s|i0) * P(l|g2)
# + P(g0|d1,i1) * P(d1) * P(i1) * P(s|i1) * P(l|g0)
# + P(g1|d1,i1) * P(d1) * P(i1) * P(s|i1) * P(l|g1)
# + P(g2|d1,i1) * P(d1) * P(i1) * P(s|i1) * P(l|g2)
# =>
# P(l0,s0)
# = 0.3 * 0.6 * 0.7 * P(s0|i0) * P(l0|g0)
# + 0.4 * 0.6 * 0.7 * P(s0|i0) * P(l0|g1)
# + 0.3 * 0.6 * 0.7 * P(s0|i0) * P(l0|g2)
# + 0.9 * 0.6 * 0.3 * P(s0|i1) * P(l0|g0)
# + 0.08 * 0.6 * 0.3 * P(s0|i1) * P(l0|g1)
# + 0.02 * 0.6 * 0.3 * P(s0|i1) * P(l0|g2)
# + 0.05 * 0.4 * 0.7 * P(s0|i0) * P(l0|g0)
# + 0.25 * 0.4 * 0.7 * P(s0|i0) * P(l0|g1)
# + 0.7 * 0.4 * 0.7 * P(s0|i0) * P(l0|g2)
# + 0.5 * 0.4 * 0.3 * P(s0|i1) * P(l0|g0)
# + 0.3 * 0.4 * 0.3 * P(s0|i1) * P(l0|g1)
# + 0.2 * 0.4 * 0.3 * P(s0|i1) * P(l0|g2)
#
# = 0.3 * 0.6 * 0.7 * 0.95 * 0.1
# + 0.4 * 0.6 * 0.7 * 0.95 * 0.4
# + 0.3 * 0.6 * 0.7 * 0.95 * 0.99
# + 0.9 * 0.6 * 0.3 * 0.2 * 0.1
# + 0.08 * 0.6 * 0.3 * 0.2 * 0.4
# + 0.02 * 0.6 * 0.3 * 0.2 * 0.99
# + 0.05 * 0.4 * 0.7 * 0.95 * 0.1
# + 0.25 * 0.4 * 0.7 * 0.95 * 0.4
# + 0.7 * 0.4 * 0.7 * 0.95 * 0.99
# + 0.5 * 0.4 * 0.3 * 0.2 * 0.1
# + 0.3 * 0.4 * 0.3 * 0.2 * 0.4
# + 0.2 * 0.4 * 0.3 * 0.2 * 0.99 = 0.4205178
#
# P(l0,s1)
# = 0.3 * 0.6 * 0.7 * P(s1|i0) * P(l0|g0)
# + 0.4 * 0.6 * 0.7 * P(s1|i0) * P(l0|g1)
# + 0.3 * 0.6 * 0.7 * P(s1|i0) * P(l0|g2)
# + 0.9 * 0.6 * 0.3 * P(s1|i1) * P(l0|g0)
# + 0.08 * 0.6 * 0.3 * P(s1|i1) * P(l0|g1)
# + 0.02 * 0.6 * 0.3 * P(s1|i1) * P(l0|g2)
# + 0.05 * 0.4 * 0.7 * P(s1|i0) * P(l0|g0)
# + 0.25 * 0.4 * 0.7 * P(s1|i0) * P(l0|g1)
# + 0.7 * 0.4 * 0.7 * P(s1|i0) * P(l0|g2)
# + 0.5 * 0.4 * 0.3 * P(s1|i1) * P(l0|g0)
# + 0.3 * 0.4 * 0.3 * P(s1|i1) * P(l0|g1)
# + 0.2 * 0.4 * 0.3 * P(s1|i1) * P(l0|g2)
#
# = 0.3 * 0.6 * 0.7 * 0.05 * 0.1
# + 0.4 * 0.6 * 0.7 * 0.05 * 0.4
# + 0.3 * 0.6 * 0.7 * 0.05 * 0.99
# + 0.9 * 0.6 * 0.3 * 0.8 * 0.1
# + 0.08 * 0.6 * 0.3 * 0.8  * 0.4
# + 0.02 * 0.6 * 0.3 * 0.8  * 0.99
# + 0.05 * 0.4 * 0.7 * 0.05 * 0.1
# + 0.25 * 0.4 * 0.7 * 0.05 * 0.4
# + 0.7 * 0.4 * 0.7 * 0.05 * 0.99
# + 0.5 * 0.4 * 0.3 * 0.8  * 0.1
# + 0.3 * 0.4 * 0.3 * 0.8  * 0.4
# + 0.2 * 0.4 * 0.3 * 0.8 * 0.99 = 0.0771462
#
# P(l1,s0)
# = 0.3 * 0.6 * 0.7 * P(s0|i0) * P(l1|g0)
# + 0.4 * 0.6 * 0.7 * P(s0|i0) * P(l1|g1)
# + 0.3 * 0.6 * 0.7 * P(s0|i0) * P(l1|g2)
# + 0.9 * 0.6 * 0.3 * P(s0|i1) * P(l1|g0)
# + 0.08 * 0.6 * 0.3 * P(s0|i1) * P(l1|g1)
# + 0.02 * 0.6 * 0.3 * P(s0|i1) * P(l1|g2)
# + 0.05 * 0.4 * 0.7 * P(s0|i0) * P(l1|g0)
# + 0.25 * 0.4 * 0.7 * P(s0|i0) * P(l1|g1)
# + 0.7 * 0.4 * 0.7 * P(s0|i0) * P(l1|g2)
# + 0.5 * 0.4 * 0.3 * P(s0|i1) * P(l1|g0)
# + 0.3 * 0.4 * 0.3 * P(s0|i1) * P(l1|g1)
# + 0.2 * 0.4 * 0.3 * P(s0|i1) * P(l1|g2)
#
# = 0.3 * 0.6 * 0.7 * 0.95 * 0.9
# + 0.4 * 0.6 * 0.7 * 0.95  * 0.6
# + 0.3 * 0.6 * 0.7 * 0.95  * 0.01
# + 0.9 * 0.6 * 0.3 * 0.2 * 0.9
# + 0.08 * 0.6 * 0.3 * 0.2 * 0.6
# + 0.02 * 0.6 * 0.3 * 0.2 * 0.01
# + 0.05 * 0.4 * 0.7 * 0.95  * 0.9
# + 0.25 * 0.4 * 0.7 * 0.95  * 0.6
# + 0.7 * 0.4 * 0.7 * 0.95  * 0.01
# + 0.5 * 0.4 * 0.3 * 0.2 * 0.9
# + 0.3 * 0.4 * 0.3 * 0.2 * 0.6
# + 0.2 * 0.4 * 0.3 * 0.2 * 0.01 = 0.3044822
#
# P(l1,s1)
# = 0.3 * 0.6 * 0.7 * 0.05 * 0.9
# + 0.4 * 0.6 * 0.7 * 0.05 * 0.6
# + 0.3 * 0.6 * 0.7 * 0.05 * 0.01
# + 0.9 * 0.6 * 0.3 * 0.8 * 0.9
# + 0.08 * 0.6 * 0.3 * 0.8 * 0.6
# + 0.02 * 0.6 * 0.3 * 0.8 * 0.01
# + 0.05 * 0.4 * 0.7 * 0.05 * 0.9
# + 0.25 * 0.4 * 0.7 * 0.05 * 0.6
# + 0.7 * 0.4 * 0.7 * 0.05 * 0.01
# + 0.5 * 0.4 * 0.3 * 0.8 * 0.9
# + 0.3 * 0.4 * 0.3 * 0.8 * 0.6
# + 0.2 * 0.4 * 0.3 * 0.8 * 0.01 = 0.1978538
assert 0.4205178 / (1 + eps) <= pd('l0', 's0') <= 0.4205178 * (1 + eps)
assert 0.0771462 / (1 + eps) <= pd('l0', 's1') <= 0.0771462 * (1 + eps)
assert 0.3044822 / (1 + eps) <= pd('l1', 's0') <= 0.3044822 * (1 + eps)
assert 0.1978538 / (1 + eps) <= pd('l1', 's1') <= 0.1978538 * (1 + eps)
