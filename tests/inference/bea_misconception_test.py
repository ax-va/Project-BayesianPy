from pyb4ml.algorithms import BEA
from pyb4ml.models import Misconception

model = Misconception()
alice = model.get_variable('Alice')
bob = model.get_variable('Bob')
charles = model.get_variable('Charles')
debbie = model.get_variable('Debbie')

eps = 1e-12

algorithm = BEA(model)
algorithm.set_query(alice, bob)
algorithm.set_evidence((charles, 'c0'), (debbie, 'd0'))
algorithm.set_elimination_order([charles, debbie])
algorithm.run()
pd = algorithm.pd
algorithm.print_pd()
# P(a,b|c0,d0)
# = f_ab(a,b) * f_bc(b,c0) * f_cd(c0,d0) * f_da(d0,a)
# / (\sum_{a, b} f_ab(a,b) * f_bc(b,c0) * f_cd(c0,d0) * f_da(d0,a))
#
# f_ab(a0,b0) * f_bc(b0,c0) * f_cd(c0,d0) * f_da(d0,a0)
# = 30 * 100 * 1 * 100 = 300000
#
# f_ab(a0,b1) * f_bc(b1,c0) * f_cd(c0,d0) * f_da(d0,a0)
# = 5 * 1 * 1 * 100 = 500
#
# f_ab(a1,b0) * f_bc(b0,c0) * f_cd(c0,d0) * f_da(d0,a1)
# = 1 * 100 * 1 * 1 = 100
#
# f_ab(a1,b1) * f_bc(b1,c0) * f_cd(c0,d0) * f_da(d0,a1)
# = 10 * 1 * 1 * 1 = 10
# =>
# (\sum_{a, b} f_ab(a,b) * f_bc(b,c0) * f_cd(c0,d0) * f_da(d0,a))
# = 300000 + 500 + 100 + 10 = 300610
# =>
# P(a0,b0|c0,d0) = 300000 / 300610 = 0.9979707927214664
# P(a0,b1|c0,d0) = 500 / 300610 = 0.0016632846545357773
# P(a1,b0|c0,d0) = 100 / 300610 = 0.0003326569309071555
# P(a1,b1|c0,d0) = 10 / 300610 = 3.3265693090715545e-05
assert 0.9979707927214664 / (1 + eps) <= pd('a0', 'b0') <= 0.9979707927214664 * (1 + eps)
assert 0.0016632846545357773 / (1 + eps) <= pd('a0', 'b1') <= 0.0016632846545357773 * (1 + eps)
assert 0.0003326569309071555 / (1 + eps) <= pd('a1', 'b0') <= 0.0003326569309071555 * (1 + eps)
assert 3.3265693090715545e-05 / (1 + eps) <= pd('a1', 'b1') <= 3.3265693090715545e-05 * (1 + eps)
