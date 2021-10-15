# PyB4ML
PyB4ML is a collection of algorithms and models written in Python for probabilistic programming. The main focus of the package is on Bayesian reasoning by using Bayesian and Markov networks. 

The collection contains the following algorithms and models.

Factored-inference related algorithms for probabilistic graphical models:
- Belief Propagation Algorithm (BPA) [1] for inference in trees (pb4ml/inference/factored/belief_propagation.py)
- Bucket Elimination Algorithm (BEA) [1] for inference in loopy graphs or computing the joint probability distribution of several query variables (pb4ml/inference/factored/bucket_elimination.py)
- Greedy Ordering Algorithm (GOA) [2] for finding an near-optimal elimination ordering by using greedy search with cost criterion "min-fill" or "weighted-min-fill" [2]

Academic probabilistic models in factor graph representation:
- Bayesian network "Student" [2] (pb4ml/models/factor_graphs/student.py)
- Markov network "Misconception" [2] (pb4ml/models/factor_graphs/misconception.py)

See the use of algorithms in the tests.

© 2021 Alexander Vasiliev

References:
[1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press, 2012;
[2] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and Techniques", MIT Press, 2009
