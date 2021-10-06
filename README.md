# PyB4ML
PyB4ML is a free open-source collection of algorithms in Bayesian machine learning written in Python for Bayesian reasoning and probabilistic programming. Bayesian and Markov networks are used here as probabilistic models.

At the moment, the collection contains the following inference algorithms:
- Belief Propagation Algorithm (BPA) (pb4ml/algorithms/inference/belief_propagation.py)
- Bucket Elimination Algorithm (BEA) (pb4ml/algorithms/inference/bucket_elimination.py)

and the following probabilistic models in factor graph representation:
- Bayesian network "Student" [2] (pb4ml/models/factor_graphs/student.py)
- Markov network "Misconception" [2] (pb4ml/models/factor_graphs/misconception.py)

See the use of algorithms in the tests.

© 2021 Alexander Vasiliev

References:

[2] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and Techniques", MIT Press, 2009
