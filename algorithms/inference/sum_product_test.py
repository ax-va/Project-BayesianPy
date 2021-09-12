from pyb4ml.algorithms.inference.sum_product import SumProduct
from pyb4ml.models.factor_graphs.student import Student

# Test the Sum-Product Algorithm on the Student model
model = Student()
algorithm = SumProduct(model)

eps = 1 / 1e15

for query in model.variables:
    print('query:', query)
    algorithm.set_query(query)
    # algorithm.run(print_messages=True, print_loop_passing=True)
    algorithm.run()
    print('-'*20)
    print('probability distribution:')
    for value in query.domain:
        print(f'P({query}={value!r})={algorithm.pd(value)}')
    # Print also the probability distribution as above: algorithm.print_pd()
    print('-'*20)
    print('-'*20)

for query in model.variables:
    algorithm.set_query(query)
    algorithm.run()
    pd = algorithm.pd
    if query.name == 'Difficulty':
        assert 0.6 - eps <= pd('d0') <= 0.6 + eps
        assert 0.4 - eps <= pd('d1') <= 0.4 + eps
    if query.name == 'Intelligence':
        assert 0.7 - eps <= pd('i0') <= 0.7 + eps
        assert 0.3 - eps <= pd('i1') <= 0.3 + eps
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
        assert 0.362 - eps <= pd('g0') <= 0.362 + eps
        # P(g1)
        # = 0.4 * 0.6 * 0.7
        # + 0.08 * 0.6 * 0.3
        # + 0.25 * 0.4 * 0.7
        # + 0.3 * 0.4 * 0.3 = 0.2884
        assert 0.2884 - eps <= pd('g1') <= 0.2884 + eps
        # P(g2)
        # = 0.3 * 0.6 * 0.7
        # + 0.02 * 0.6 * 0.3
        # + 0.7 * 0.4 * 0.7
        # + 0.2 * 0.4 * 0.3 = 0.3496
        assert 0.3496 - eps <= pd('g2') <= 0.3496 + eps
    if query.name == 'SAT':
        # P(s) = \sum_{i} {P(s|i) * P(i)}
        # = P(s|i0) * P(i0) + P(s|i1) * P(i1)
        # = P(s|i0) * 0.7 + P(s|i1) * 0.3
        # =>
        # P(s0) = 0.95 * 0.7 + 0.2 * 0.3 = 0.725
        assert 0.725 - eps <= pd('s0') <= 0.725 + eps
        # P(s1) = 0.05 * 0.7 + 0.8 * 0.3 = 0.275
        assert 0.275 - eps <= pd('s1') <= 0.275 + eps
    if query.name == 'Letter':
        # P(l) = \sum_{g} {P(l|g) * P(g)}
        # = P(l|g0) * P(g0) + P(l|g1) * P(g1) + P(l|g2) * P(g2)
        # = P(l|g0) * 0.362 + P(l|g1) * 0.2884 + P(l|g2) * 0.3496
        # =>
        # P(l0) = 0.1 * 0.362 + 0.4 * 0.2884 + 0.99 * 0.3496 = 0.497664
        assert 0.497664- eps <= pd('l0') <= 0.497664 + eps
        # P(l1) = 0.9 * 0.362 + 0.6 * 0.2884 + 0.01 * 0.3496 = 0.502336
        assert 0.502336 - eps <= pd('l1') <= 0.502336 + eps
