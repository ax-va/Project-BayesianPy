import pyb4ml.modeling.utils.names as names

from pyb4ml.modeling.categorical.variable import Variable


def check_variable_instance(variable):
    if not isinstance(variable, Variable):
        raise ValueError(f"Object '{variable}' is not an instance of class {Variable.__class__.__name__}")


def check_is_variable_evidential(variable):
    if not variable.is_evidential():
        raise ValueError(f"Variable '{variable.name}' is not evidential. "
                         f"An evidential variable must have only one value in its domain.")


def check_are_variables_evidential(variables):
    for var in variables:
        check_is_variable_evidential(var)


def check_is_variable_non_evidential(variable):
    if variable.is_evidential():
        raise ValueError(f"Variable '{variable.name}' is not non-evidential. "
                         f"A non-evidential variable must have more than one value in its domain.")


def check_disjoint(elements1, elements2):
    set1 = set(elements1)
    set2 = set(elements2)
    if not set1.isdisjoint(set2):
        raise ValueError(f"Elements {names.get_names(elements1)} and {names.get_names(elements2)} must be disjoint")


