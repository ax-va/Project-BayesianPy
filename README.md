# PyB4ML
PyB4ML is a collection of algorithms and models written in Python for probabilistic programming. The main focus of the package is on Bayesian reasoning by using Bayesian or Markov networks. 

The collection contains the following algorithms and models.

Factored-inference-related algorithms for probabilistic graphical models:
- Belief Propagation (BP) [1] for efficient inference in trees (pb4ml/inference/factored/belief_propagation.py)
- Bucket Elimination (BE) [1] for inference in loopy graphs or computing the joint probability distribution of several query variables (pb4ml/inference/factored/bucket_elimination.py)
- Greedy Ordering (GO) [2] for greedy search for a near-optimal variable elimination ordering (pb4ml/inference/factored/greedy_ordering.py)

Academic probabilistic models in factor graph representation:
- Bayesian network "Extended Student" [2] (pb4ml/models/factor_graphs/extended_student.py)
- Bayesian network "Student" [2] (pb4ml/models/factor_graphs/student.py)
- Markov network "Misconception" [2] (pb4ml/models/factor_graphs/misconception.py)

See the use of algorithms in the tests.

© 2021 Alexander Vasiliev

References:
[1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press, 2012;
[2] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and Techniques", MIT Press, 2009
