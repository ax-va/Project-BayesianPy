from pyb4ml.inference.factored.greedy_ordering import GO
from pyb4ml.models.factor_graphs.extended_student import ExtendedStudent

model = ExtendedStudent()
job = model.get_variable('Job')

algorithm = GO(model)
algorithm.set_query(job)
algorithm.run(print_info=True)