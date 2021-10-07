# PyB4ML
PyB4ML is a collection of algorithms and models in Bayesian machine learning written in Python for probabilistic programming. The main focus of the package is on Bayesian reasoning by using Bayesian and Markov networks. 

At the moment, the collection contains the following inference algorithms:
- Belief Propagation Algorithm (BPA) [1] (pb4ml/algorithms/inference/belief_propagation.py)
- Bucket Elimination Algorithm (BEA) [1] (pb4ml/algorithms/inference/bucket_elimination.py)

and the following probabilistic models in factor graph representation:
- Bayesian network "Student" [2] (pb4ml/models/factor_graphs/student.py)
- Markov network "Misconception" [2] (pb4ml/models/factor_graphs/misconception.py)

See the use of algorithms in the tests.

© 2021 Alexander Vasiliev

References:
[1] David Barber, "Bayesian Reasoning and Machine Learning", Cambridge University Press, 2012;
[2] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and Techniques", MIT Press, 2009
