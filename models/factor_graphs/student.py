from pyb4ml.modeling import Factor
from pyb4ml.modeling import Variable
from pyb4ml.modeling import FactorGraph


class Student(FactorGraph):
    """
    The Student model [1, page 53]. See also "student.pdf" or "student.odp".

    [1] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models:
    Principles and Techniques", MIT Press, 2009
    """
    def __init__(self):
        # Random variables
        difficulty = Variable(domain={'d0', 'd1'}, name='Difficulty')
        intelligence = Variable(domain={'i0', 'i1'}, name='Intelligence')
        grade = Variable(domain={'g0', 'g1', 'g2'}, name='Grade')
        sat = Variable(domain={'s0', 's1'}, name='SAT')
        letter = Variable(domain={'l0', 'l1'}, name='Letter')
        variables = {
            difficulty,
            intelligence,
            grade,
            sat,
            letter
        }

        # Conditional probability distributions (CPDs)
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

        # Factors
        f0 = Factor(
            variables=(difficulty, ),
            function=lambda x: cpd_difficulty[x],
            name='f0'
        )
        f1 = Factor(
            variables=(intelligence, ),
            function=lambda x: cpd_intelligence[x],
            name='f1'
        )
        f2 = Factor(
            variables=(
                difficulty,
                intelligence,
                grade
            ),
            function=lambda x, y, z: cpd_grade[(x, y)][z],
            name='f2'
        )
        f3 = Factor(
            variables=(
                intelligence,
                sat
            ),
            function=lambda x, y: cpd_sat[x][y],
            name='f3'
        )
        f4 = Factor(
            variables=(
                grade,
                letter
            ),
            function=lambda x, y: cpd_letter[x][y],
            name='f4'
        )
        factors = {
            f0,
            f1,
            f2,
            f3,
            f4
        }
        FactorGraph.__init__(self, factors, variables)
