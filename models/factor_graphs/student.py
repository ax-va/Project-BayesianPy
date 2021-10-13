"""
The module contains the class of the Student model.

Attention:  The author is not responsible for any damage that can be caused by the use
of this code.  You use this code at your own risk.  Any claim against the author is 
legally void.  By using this code, you agree to the terms imposed by the author.

Achtung:  Der Autor haftet nicht für Schäden, die durch die Verwendung dieses Codes
entstehen können.  Sie verwenden dieses Code auf eigene Gefahr.  Jegliche Ansprüche 
gegen den Autor sind rechtlich nichtig.  Durch die Verwendung dieses Codes stimmen 
Sie den vom Autor auferlegten Bedingungen zu.

© 2021 Alexander Vasiliev
"""
from pyb4ml.modeling import Factor, FactorGraph
from pyb4ml.modeling.categorical.variable import Variable


class Student(FactorGraph):
    """
    Implements the Student model [1] that is a Bayesian network with tabular probability 
    distributions and without loops in the graph.  See also "student.pdf" or "student.odp" in
    this directory.
    
    References:

    [1] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and 
    Techniques", MIT Press, 2009
    """
    def __init__(self):
        # Create random variables
        difficulty = Variable(domain={'d0', 'd1'}, name='Difficulty')
        intelligence = Variable(domain={'i0', 'i1'}, name='Intelligence')
        grade = Variable(domain={'g0', 'g1', 'g2'}, name='Grade')
        sat = Variable(domain={'s0', 's1'}, name='SAT')
        letter = Variable(domain={'l0', 'l1'}, name='Letter')

        # Create conditional probability distributions (CPDs)
        cpd_difficulty = {
            'd0': 0.6,
            'd1': 0.4
        }
        cpd_intelligence = {
            'i0': 0.7,
            'i1': 0.3
        }
        cpd_grade = {
            ('d0', 'i0'): {'g0': 0.30, 'g1': 0.40, 'g2': 0.30},
            ('d0', 'i1'): {'g0': 0.90, 'g1': 0.08, 'g2': 0.02},
            ('d1', 'i0'): {'g0': 0.05, 'g1': 0.25, 'g2': 0.70},
            ('d1', 'i1'): {'g0': 0.50, 'g1': 0.30, 'g2': 0.20}
        }
        cpd_sat = {
            'i0': {'s0': 0.95, 's1': 0.05},
            'i1': {'s0': 0.20, 's1': 0.80}
        }
        cpd_letter = {
            'g0': {'l0': 0.10, 'l1': 0.90},
            'g1': {'l0': 0.40, 'l1': 0.60},
            'g2': {'l0': 0.99, 'l1': 0.01}
        }

        # Create factors
        f0 = Factor(
            variables=(difficulty, ),
            function=lambda d: cpd_difficulty[d],
            name='f0'
        )
        f1 = Factor(
            variables=(intelligence, ),
            function=lambda i: cpd_intelligence[i],
            name='f1'
        )
        f2 = Factor(
            variables=(
                difficulty,
                intelligence,
                grade
            ),
            function=lambda d, i, g: cpd_grade[(d, i)][g],
            name='f2'
        )
        f3 = Factor(
            variables=(
                intelligence,
                sat
            ),
            function=lambda i, s: cpd_sat[i][s],
            name='f3'
        )
        f4 = Factor(
            variables=(
                grade,
                letter
            ),
            function=lambda g, l: cpd_letter[g][l],
            name='f4'
        )

        # Create a factorization
        factors = {
            f0,
            f1,
            f2,
            f3,
            f4
        }

        # Create a factor graph from the factors
        FactorGraph.__init__(self, factors)
