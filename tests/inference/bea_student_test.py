from pyb4ml.algorithms import BEA
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
algorithm.set_evidence((letter, 'l0'), (sat, 's0'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.4742196406430358 - eps <= pd('d0') <= 0.4742196406430358 + eps
assert 0.5257803593569642 - eps <= pd('d1') <= 0.5257803593569642 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.3972483414607588 - eps <= pd('d0') <= 0.3972483414607588 + eps
assert 0.6027516585392411 - eps <= pd('d1') <= 0.6027516585392411 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's0'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.7737141941302315 - eps <= pd('d0') <= 0.7737141941302315 + eps
assert 0.22628580586976843 - eps <= pd('d1') <= 0.22628580586976843 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's1'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.6790559493929356 - eps <= pd('d0') <= 0.6790559493929356 + eps
assert 0.32094405060706443 - eps <= pd('d1') <= 0.32094405060706443 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.4622878086419753 - eps <= pd('d0') <= 0.4622878086419753 + eps
assert 0.5377121913580247 - eps <= pd('d1') <= 0.5377121913580247 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.7364313925340807 - eps <= pd('d0') <= 0.7364313925340807 + eps
assert 0.26356860746591926 - eps <= pd('d1') <= 0.26356860746591926 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's0'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.6 - eps <= pd('d0') <= 0.6 + eps
assert 0.4 - eps <= pd('d1') <= 0.4 + eps

algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's1'))
algorithm.set_order([letter, sat, intelligence, grade])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.6 - eps <= pd('d0') <= 0.6 + eps
assert 0.4 - eps <= pd('d1') <= 0.4 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l0'), (sat, 's0'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.07627202787313875 - eps <= pd('g0') <= 0.07627202787313875 + eps
assert 0.32590872028474893 - eps <= pd('g1') <= 0.32590872028474893 + eps
assert 0.5978192518421124 - eps <= pd('g2') <= 0.5978192518421124 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l0'), (sat, 's1'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.4434481273371577 - eps <= pd('g0') <= 0.4434481273371577 + eps
assert 0.2599996084343245 - eps <= pd('g1') <= 0.2599996084343245 + eps
assert 0.29655226422851777 - eps <= pd('g2') <= 0.29655226422851777 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l1'), (sat, 's0'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.5810710656407828 - eps <= pd('g0') <= 0.5810710656407828 + eps
assert 0.4138173427364206 - eps <= pd('g1') <= 0.4138173427364206 + eps
assert 0.005111591622796632 - eps <= pd('g2') <= 0.005111591622796632 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd0'), (letter, 'l1'), (sat, 's1'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.9103575782746748 - eps <= pd('g0') <= 0.9103575782746748 + eps
assert 0.08895915113677469 - eps <= pd('g1') <= 0.08895915113677469 + eps
assert 0.0006832705885505288 - eps <= pd('g2') <= 0.0006832705885505288 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l0'), (sat, 's0'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.011442786069651743 - eps <= pd('g0') <= 0.011442786069651743 + eps
assert 0.13333333333333333 - eps <= pd('g1') <= 0.13333333333333333 + eps
assert 0.8552238805970149 - eps <= pd('g2') <= 0.8552238805970149 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l0'), (sat, 's1'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.10473118279569894 - eps <= pd('g0') <= 0.10473118279569894 + eps
assert 0.2778494623655915 - eps <= pd('g1') <= 0.2778494623655915 + eps
assert 0.6174193548387096 - eps <= pd('g2') <= 0.6174193548387096 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l1'), (sat, 's0'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.33047895500725694 - eps < pd('g0') < 0.33047895500725694 + eps
assert 0.641799709724238 - eps < pd('g1') < 0.641799709724238 + eps
assert 0.027721335268505072 - eps < pd('g2') < 0.027721335268505072 + eps

algorithm.set_query(grade)
algorithm.set_evidence((difficulty, 'd1'), (letter, 'l1'), (sat, 's1'))
algorithm.set_order([difficulty, letter, sat, intelligence])
algorithm.run()
pd = algorithm.pd
# The values for the assertions are obtained using the BPA
assert 0.690236220472441 - eps <= pd('g0') <= 0.690236220472441 + eps
assert 0.30519685039370076 - eps <= pd('g1') <= 0.30519685039370076 + eps
assert 0.004566929133858268 - eps <= pd('g2') <= 0.004566929133858268 + eps
