if __name__ == '__main__':
    import sys
    # Add the path of '..\\pyb4ml' to sys.path
    # in the case of executing from '..\\pyb4ml\\tests\\inference'.
    # Otherwise, the package 'pyb4ml' will not be found.
    if '..\\' not in sys.path:
        sys.path.insert(0, '..\\')

from pyb4ml.inference import BP
from pyb4ml.models import Student

# Test the Belief Propagation algorithm on the Student model
model = Student()
algorithm = BP(model)  # Belief Propagation algorithm

eps = 1e-10

# Test marginal distributions

for query in model.variables:
    algorithm.set_query(query)
    algorithm.run()
    algorithm.print_pd()
    pd = algorithm.pd
    if query.name == 'Difficulty':
        assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
        assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)
    if query.name == 'Intelligence':
        assert 0.7 / (1 + eps) <= pd('i0') <= 0.7 * (1 + eps)
        assert 0.3 / (1 + eps) <= pd('i1') <= 0.3 * (1 + eps)
    if query.name == 'Grade':
        # P(g) = \sum_{d,i,s,l} {P(l|g) * P(g|d,i) * P(s|i) * P(d) * P(i)}
        # % \sum_{s} {P(s|i) * P(i)} = P(i)
        # % \sum_{l} {P(l|g)} = 1
        # = P(g|d0,i0) * P(d0) * P(i0)
        # + P(g|d0,i1) * P(d0) * P(i1)
        # + P(g|d1,i0) * P(d1) * P(i0)
        # + P(g|d1,i1) * P(d1) * p(i1)
        # = P(g|d0,i0) * 0.6 * 0.7
        # + P(g|d0,i1) * 0.6 * 0.3
        # + P(g|d1,i0) * 0.4 * 0.7
        # + P(g|d1,i1) * 0.4 * 0.3
        # =>
        # P(g0)
        # = 0.3 * 0.6 * 0.7
        # + 0.9 * 0.6 * 0.3
        # + 0.05 * 0.4 * 0.7
        # + 0.5 * 0.4 * 0.3 = 0.362
        assert 0.362 / (1 + eps) <= pd('g0') <= 0.362 * (1 + eps)
        # P(g1)
        # = 0.4 * 0.6 * 0.7
        # + 0.08 * 0.6 * 0.3
        # + 0.25 * 0.4 * 0.7
        # + 0.3 * 0.4 * 0.3 = 0.2884
        assert 0.2884 / (1 + eps) <= pd('g1') <= 0.2884 * (1 + eps)
        # P(g2)
        # = 0.3 * 0.6 * 0.7
        # + 0.02 * 0.6 * 0.3
        # + 0.7 * 0.4 * 0.7
        # + 0.2 * 0.4 * 0.3 = 0.3496
        assert 0.3496 / (1 + eps) <= pd('g2') <= 0.3496 * (1 + eps)
    if query.name == 'SAT':
        # P(s) = \sum_{i} {P(s|i) * P(i)}
        # = P(s|i0) * P(i0) + P(s|i1) * P(i1)
        # = P(s|i0) * 0.7 + P(s|i1) * 0.3
        # =>
        # P(s0) = 0.95 * 0.7 + 0.2 * 0.3 = 0.725
        assert 0.725 / (1 + eps) <= pd('s0') <= 0.725 * (1 + eps)
        # P(s1) = 0.05 * 0.7 + 0.8 * 0.3 = 0.275
        assert 0.275 / (1 + eps) <= pd('s1') <= 0.275 * (1 + eps)
    if query.name == 'Letter':
        # P(l) = \sum_{g} {P(l|g) * P(g)}
        # = P(l|g0) * P(g0) + P(l|g1) * P(g1) + P(l|g2) * P(g2)
        # = P(l|g0) * 0.362 + P(l|g1) * 0.2884 + P(l|g2) * 0.3496
        # =>
        # P(l0) = 0.1 * 0.362 + 0.4 * 0.2884 + 0.99 * 0.3496 = 0.497664
        assert 0.497664 / (1 + eps) <= pd('l0') <= 0.497664 * (1 + eps)
        # P(l1) = 0.9 * 0.362 + 0.6 * 0.2884 + 0.01 * 0.3496 = 0.502336
        assert 0.502336 / (1 + eps) <= pd('l1') <= 0.502336 * (1 + eps)

# Test conditional distributions

# P(d,l0,s0) = P(d) * (
# P(i0) * P(s0|i0) * (P(g0|d,i0) * P(l0|g0) + P(g1|d,i0) * P(l0|g1) + P(g2|d,i0) * P(l0|g2)) +
# P(i1) * P(s0|i1) * (P(g0|d,i1) * P(l0|g0) + P(g1|d,i1) * P(l0|g1) + P(g2|d,i1) * P(l0|g2)))
# =>
# P(d0,l0,s0) = 0.6 * (
# 0.7 * 0.95 * (0.3 * 0.1 + 0.4 * 0.4 + 0.3 * 0.99) +
# 0.3 * 0.2 * (0.9 * 0.1 + 0.08 * 0.4 + 0.02 * 0.99)) = 0.1994178
# P(d1,l0,s0) = 0.4 * (
# 0.7 * 0.95 * (0.05 * 0.1 + 0.25 * 0.4 + 0.7 * 0.99) +
# 0.3 * 0.2 * (0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.99)) = 0.2211
# =>
# P(l0,s0) = 0.1994178 + 0.2211 = 0.4205178
# =>
# P(d0|l0,s0) = 0.1994178 / 0.4205178 = 0.474219640643
# P(d1|l0,s0) = 0.2211 / 0.4205178 = 0.525780359357
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's0'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.474219640643 / (1 + eps) <= pd('d0') <= 0.474219640643 * (1 + eps)
assert 0.525780359357 / (1 + eps) <= pd('d1') <= 0.525780359357 * (1 + eps)

# P(d,l0,s1) = P(d) * (
# P(i0) * P(s1|i0) * (P(g0|d,i0) * P(l0|g0) + P(g1|d,i0) * P(l0|g1) + P(g2|d,i0) * P(l0|g2)) +
# P(i1) * P(s1|i1) * (P(g0|d,i1) * P(l0|g0) + P(g1|d,i1) * P(l0|g1) + P(g2|d,i1) * P(l0|g2)))
# =>
# P(d0,l0,s1) = 0.6 * (
# 0.7 * 0.05 * (0.3 * 0.1 + 0.4 * 0.4 + 0.3 * 0.99) +
# 0.3 * 0.8 * (0.9 * 0.1 + 0.08 * 0.4 + 0.02 * 0.99)) = 0.0306462
# P(d1,l0,s1) = 0.4 * (
# 0.7 * 0.05 * (0.05 * 0.1 + 0.25 * 0.4 + 0.7 * 0.99) +
# 0.3 * 0.8 * (0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.99)) = 0.0465
# =>
# P(l0,s1) = 0.0306462 + 0.0465 = 0.0771462
# =>
# P(d0|l0,s1) = 0.0306462 / 0.0771462 = 0.397248341461
# P(d1|l0,s1) = 0.0465 / 0.0771462 = 0.602751658539
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's1'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.397248341461 / (1 + eps) <= pd('d0') <= 0.397248341461 * (1 + eps)
assert 0.602751658539 / (1 + eps) <= pd('d1') <= 0.602751658539 * (1 + eps)

# P(d,l1,s0) = P(d) * (
# P(i0) * P(s0|i0) * (P(g0|d,i0) * P(l1|g0) + P(g1|d,i0) * P(l1|g1) + P(g2|d,i0) * P(l1|g2)) +
# P(i1) * P(s0|i1) * (P(g0|d,i1) * P(l1|g0) + P(g1|d,i1) * P(l1|g1) + P(g2|d,i1) * P(l1|g2)))
# =>
# P(d0,l1,s0) = 0.6 * (
# 0.7 * 0.95 * (0.3 * 0.9 + 0.4 * 0.6 + 0.3 * 0.01) +
# 0.3 * 0.2 * (0.9 * 0.9 + 0.08 * 0.6 + 0.02 * 0.01)) = 0.2355822
# P(d1,l1,s0) = 0.4 * (
# 0.7 * 0.95 * (0.05 * 0.9 + 0.25 * 0.6 + 0.7 * 0.01) +
# 0.3 * 0.2 * (0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.01)) = 0.0689
# =>
# P(l1,s0) = 0.2355822 + 0.0689 = 0.3044822
# =>
# P(d0|l1,s0) = 0.2355822 / 0.3044822 = 0.77371419413
# P(d1|l1,s0) = 0.0689 / 0.3044822 = 0.22628580587
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's0'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.77371419413 / (1 + eps) <= pd('d0') <= 0.77371419413 * (1 + eps)
assert 0.22628580587 / (1 + eps) <= pd('d1') <= 0.22628580587 * (1 + eps)

# P(d,l1,s1) = P(d) * (
# P(i0) * P(s1|i0) * (P(g0|d,i0) * P(l1|g0) + P(g1|d,i0) * P(l1|g1) + P(g2|d,i0) * P(l1|g2)) +
# P(i1) * P(s1|i1) * (P(g0|d,i1) * P(l1|g0) + P(g1|d,i1) * P(l1|g1) + P(g2|d,i1) * P(l1|g2)))
# =>
# P(d0,l1,s1) = 0.6 * (
# 0.7 * 0.05 * (0.3 * 0.9 + 0.4 * 0.6 + 0.3 * 0.01) +
# 0.3 * 0.8 * (0.9 * 0.9 + 0.08 * 0.6 + 0.02 * 0.01)) = 0.1343538
# P(d1,l1,s1) = 0.4 * (
# 0.7 * 0.05 * (0.05 * 0.9 + 0.25 * 0.6 + 0.7 * 0.01) +
# 0.3 * 0.8 * (0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.01)) = 0.0635
# =>
# P(l1,s1) = 0.1343538 + 0.0635 = 0.1978538
# =>
# P(d0|l1,s1) = 0.1343538 / 0.1978538 = 0.679055949393
# P(d1|l1,s1) = 0.0635 / 0.1978538 = 0.320944050607
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'), (sat, 's1'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.679055949393 / (1 + eps) <= pd('d0') <= 0.679055949393 * (1 + eps)
assert 0.320944050607 / (1 + eps) <= pd('d1') <= 0.320944050607 * (1 + eps)

# P(d0,l0) = 0.1994178 + 0.0306462= 0.230064
# P(d1,l0) = 0.2211 + 0.0465 = 0.2676
# P(l0) = 0.230064 + 0.2676 = 0.497664
# P(l0) = 0.4205178 + 0.0771462 = 0.497664
# =>
# P(d0|l0) = 0.230064 / 0.497664 = 0.462287808642
# P(d1|l0) = 0.2676 / 0.497664 = 0.537712191358
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.462287808642 / (1 + eps) <= pd('d0') <= 0.462287808642 * (1 + eps)
assert 0.537712191358 / (1 + eps) <= pd('d1') <= 0.537712191358 * (1 + eps)

# P(d0,l1) = 0.2355822 + 0.1343538 = 0.369936
# P(d1,l1) = 0.0689 + 0.0635 = 0.1324
# P(l1) = 0.369936 + 0.1324 = 0.502336
# P(l1) = 0.3044822 + 0.1978538 = 0.502336
# =>
# P(d0|l1) = 0.369936 / 0.502336 = 0.736431392534
# P(d1|l1) = 0.1324 / 0.502336 = 0.263568607466
difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l1'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.736431392534 / (1 + eps) <= pd('d0') <= 0.736431392534 * (1 + eps)
assert 0.263568607466 / (1 + eps) <= pd('d1') <= 0.263568607466 * (1 + eps)

# P(d0,s0) = 0.1994178 + 0.2355822 = 0.435
# P(d1,s0) = 0.2211 + 0.0689 = 0.29
# P(s0) = 0.435 + 0.29 = 0.725
# P(d0|s0) = 0.435 / 0.725 = 0.6
# P(d1|s0) = 0.29 / 0.725 = 0.4
# The trail Difficulty - Grade - Intelligence - SAT
# is not active given the empty set of observed variables
# because neither Grade nor Letter is observed
difficulty = model.get_variable('Difficulty')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's0'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)

# P(d0,s1) = 0.0306462 + 0.1343538 = 0.165
# P(d1,s1) = 0.0465 + 0.0635 = 0.11
# P(s1) = = 0.275
# P(d0|s1) = 0.165 / 0.275 = 0.6
# P(d1|s1) = 0.11 / 0.275 = 0.4
# The trail Difficulty - Grade - Intelligence - SAT
# is not active given the empty set of observed variables
# because neither Grade nor Letter is observed
difficulty = model.get_variable('Difficulty')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((sat, 's1'))
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)

# algorithm.set_evidence((difficulty, 'd0'))
# algorithm.set_query(sat)

# Test marginal distributions again

algorithm.set_evidence(None)
for query in model.variables:
    algorithm.set_query(query)
    algorithm.run()
    algorithm.print_pd()
    pd = algorithm.pd
    if query.name == 'Difficulty':
        assert 0.6 / (1 + eps) <= pd('d0') <= 0.6 * (1 + eps)
        assert 0.4 / (1 + eps) <= pd('d1') <= 0.4 * (1 + eps)
    if query.name == 'Intelligence':
        assert 0.7 / (1 + eps) <= pd('i0') <= 0.7 * (1 + eps)
        assert 0.3 / (1 + eps) <= pd('i1') <= 0.3 * (1 + eps)
    if query.name == 'Grade':
        assert 0.362 / (1 + eps) <= pd('g0') <= 0.362 * (1 + eps)
        assert 0.2884 / (1 + eps) <= pd('g1') <= 0.2884 * (1 + eps)
        assert 0.3496 / (1 + eps) <= pd('g2') <= 0.3496 * (1 + eps)
    if query.name == 'SAT':
        assert 0.725 / (1 + eps) <= pd('s0') <= 0.725 * (1 + eps)
        assert 0.275 / (1 + eps) <= pd('s1') <= 0.275 * (1 + eps)
    if query.name == 'Letter':
        assert 0.497664 / (1 + eps) <= pd('l0') <= 0.497664 * (1 + eps)
        assert 0.502336 / (1 + eps) <= pd('l1') <= 0.502336 * (1 + eps)

algorithm.clear_cached_messages()

# Test conditional distributions again

difficulty = model.get_variable('Difficulty')
letter = model.get_variable('Letter')
sat = model.get_variable('SAT')
algorithm.set_query(difficulty)
algorithm.set_evidence((letter, 'l0'), (sat, 's0'))
algorithm.run(print_info=True)
pd = algorithm.pd
algorithm.print_pd()
assert 0.474219640643 / (1 + eps) <= pd('d0') <= 0.474219640643 * (1 + eps)
assert 0.525780359357 / (1 + eps) <= pd('d1') <= 0.525780359357 * (1 + eps)
