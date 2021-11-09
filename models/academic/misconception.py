"""
The module contains the class of the Misconception model.

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


class Misconception(FactorGraph):
    """
    Implements the Misconception model [1] that is a Markov network with tabular factor
    values.  See also "misconception.pdf" or "misconception.odp" in this directory.

    References:

    [1] Daphne Koller and Nir Friedman, "Probabilistic Graphical Models: Principles and
    Techniques", MIT Press, 2009
    """
    def __init__(self):
        # Create random variables
        alice = Variable(domain={'a0', 'a1'}, name='Alice')
        bob = Variable(domain={'b0', 'b1'}, name='Bob')
        charles = Variable(domain={'c0', 'c1'}, name='Charles')
        debbie = Variable(domain={'d0', 'd1'}, name='Debbie')

        # Create factor values
        dict_ab = {
            ('a0', 'b0'): 30,
            ('a0', 'b1'): 5,
            ('a1', 'b0'): 1,
            ('a1', 'b1'): 10
        }
        dict_bc = {
            ('b0', 'c0'): 100,
            ('b0', 'c1'): 1,
            ('b1', 'c0'): 1,
            ('b1', 'c1'): 100
        }
        dict_cd = {
            ('c0', 'd0'): 1,
            ('c0', 'd1'): 100,
            ('c1', 'd0'): 100,
            ('c1', 'd1'): 1
        }
        dict_da = {
            ('d0', 'a0'): 100,
            ('d0', 'a1'): 1,
            ('d1', 'a0'): 1,
            ('d1', 'a1'): 100
        }

        # Create factors
        f_ab = Factor(
            variables=(alice, bob),
            function=lambda a, b: dict_ab[(a, b)],
            name='f_ab'
        )
        f_bc = Factor(
            variables=(bob, charles),
            function=lambda b, c: dict_bc[(b, c)],
            name='f_bc'
        )
        f_cd = Factor(
            variables=(charles, debbie),
            function=lambda c, d: dict_cd[(c, d)],
            name='f_cd'
        )
        f_da = Factor(
            variables=(debbie, alice),
            function=lambda d, a: dict_da[(d, a)],
            name='f_da'
        )

        # Create a factorization
        factors = {f_ab, f_bc, f_cd, f_da}

        # Create a factor graph
        FactorGraph.__init__(self, factors)
