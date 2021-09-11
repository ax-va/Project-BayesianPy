from pyb4ml.modeling.factor_graph.factorization import Factorization
from pyb4ml.modeling.factor_graph.variable import Variable


class Student:
    """
    The Student example from "Probabilistic Graphical Models: Principles and Techniques"
    by Daphne Koller and Nir Friedman, 2009, MIT Press, page 53
    """
    def __init__(self):
        self._difficulty = Variable(domain={'d0', 'd1'}, name='Difficulty')
        self._intelligence = Variable(domain={'i0', 'i1'}, name='Intelligence')
        self._grade = Variable(domain={'g0', 'g1', 'g2'}, name='Grade')
        self._sat = Variable(domain={'s0', 's1'}, name='SAT')
        self._letter = Variable(domain={'l0', 'l1'}, name='Letter')
        self._variables = (
            self._difficulty,
            self._intelligence,
            self._grade,
            self._sat,
            self._letter,
            self._letter
        )

        self._cpd_difficulty = {
            'd0': 0.6,
            'd1': 0.4
        }
        self._cpd_intelligence = {
            'i0': 0.7,
            'i1': 0.3
        }
        self._cpd_grade = {
            ('i0', 'd0'): {'g0': 0.30, 'g1': 0.40, 'g2': 0.30},
            ('i0', 'd1'): {'g0': 0.05, 'g1': 0.25, 'g2': 0.70},
            ('i1', 'd0'): {'g0': 0.90, 'g1': 0.08, 'g2': 0.02},
            ('i1', 'd1'): {'g0': 0.50, 'g1': 0.30, 'g2': 0.20}
        }
        self._cpd_sat = {
            'i0': {'s0': 0.95, 's1': 0.05},
            'i1': {'s0': 0.20, 's1': 0.80}
        }
        self._cpd_letter = {
            'g0': {'l0': 0.10, 'l1': 0.90},
            'g1': {'l0': 0.40, 'l1': 0.60},
            'g2': {'l0': 0.99, 'l1': 0.01}
        }
        self._factors = (
            self._f_cpd_difficulty,
            self._f_cpd_intelligence,
            self._f_cpd_grade,
            self._f_cpd_sat,
            self._f_cpd_letter
        )

        self._factorization = Factorization(
            factors=self._factors,
            variables=self._variables
        )

    @property
    def factorization(self):
        return self._factorization

    def _f_cpd_difficulty(self, x):
        return self._cpd_difficulty[x]

    def _f_cpd_intelligence(self, x):
        return self._cpd_intelligence[x]

    def _f_cpd_grade(self, x, y, z):
        return self._cpd_grade[(x,y)][z]

    def _f_cpd_sat(self, x, y):
        return self._cpd_sat[x][y]

    def _f_cpd_letter(self, x, y):
        return self._cpd_letter[x][y]