"""
The module contains the class of the Extended Student model.

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
from pyb4ml.models import Student


class ExtendedStudent(Student):
    """
    Implements the Extended Student model [KF09] that is a Bayesian network with tabular
    probability distributions and is a loopy graph.  See also "extended_student.pdf"
    or "extended_student.odp" in this directory.

    References:

    [KF09] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles
    and Techniques", The MIT Press, 2009
    """
    def __init__(self):
        Student.__init__(self)

        # Get needed variables
        difficulty = self.get_variable('Difficulty')
        grade = self.get_variable('Grade')
        sat = self.get_variable('SAT')
        letter = self.get_variable('Letter')

        # Get needed factors
        f_i = self.get_factor('f_i')
        f_dig = self.get_factor('f_dig')
        f_is = self.get_factor('f_is')
        f_gl = self.get_factor('f_gl')

        # Create extending random variables
        coherence = Variable(domain={'c0', 'c1', 'c2'}, name='Coherence')
        job = Variable(domain={'j0', 'j1'}, name='Job')
        happy = Variable(domain={'h0', 'h1', 'h2'}, name='Happy')

        # Create extending CPDs
        cpd_coherence = {
            'c0': 0.2,
            'c1': 0.5,
            'c2': 0.3
        }
        cpd_difficulty = {
            'c0': {'d0': 0.2, 'd1': 0.8},
            'c1': {'d0': 0.5, 'd1': 0.5},
            'c2': {'d0': 0.8, 'd1': 0.2}
        }
        cpd_job = {
            ('l0', 's0'): {'j0': 0.95, 'j1': 0.05},
            ('l0', 's1'): {'j0': 0.25, 'j1': 0.75},
            ('l1', 's0'): {'j0': 0.65, 'j1': 0.35},
            ('l1', 's1'): {'j0': 0.15, 'j1': 0.85}
        }
        cpd_happy = {
            ('g0', 'j0'): {'h0': 0.60, 'h1': 0.30, 'h2': 0.10},
            ('g0', 'j1'): {'h0': 0.01, 'h1': 0.10, 'h2': 0.89},
            ('g1', 'j0'): {'h0': 0.80, 'h1': 0.15, 'h2': 0.05},
            ('g1', 'j1'): {'h0': 0.10, 'h1': 0.20, 'h2': 0.70},
            ('g2', 'j0'): {'h0': 0.95, 'h1': 0.04, 'h2': 0.01},
            ('g2', 'j1'): {'h0': 0.20, 'h1': 0.30, 'h2': 0.50},
        }

        # Create extending factors
        f_c = Factor(
            variables=(coherence, ),
            function=lambda c: cpd_coherence[c],
            name='f_c'
        )
        f_cd = Factor(
            variables=(coherence, difficulty),
            function=lambda c, d: cpd_difficulty[c][d],
            name='f_cd'
        )
        f_lsj = Factor(
            variables=(letter, sat, job),
            function=lambda l, s, j: cpd_job[(l, s)][j],
            name='f_lsj'
        )
        f_gjh = Factor(
            variables=(grade, job, happy),
            function=lambda g, j, h: cpd_happy[(g, j)][h],
            name='f_gjh'
        )

        # Create a factorization
        factors = {
            f_c,
            f_cd,
            f_dig,
            f_i,
            f_is,
            f_gl,
            f_lsj,
            f_gjh
        }

        # Overwrite the factor graph by the factors
        FactorGraph.__init__(self, factors)


if __name__ == '__main__':
    model = ExtendedStudent()
    for factor in model.factors:
        print(factor)
    for variable in model.variables:
        print(variable)
