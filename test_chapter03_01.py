from propositions.syntax_test import *
from propositions.semantics_test import *
from propositions.operators_test import *

def test_before_tasks(debug=False):
    assert is_binary('+'), 'Change is_binary() before testing Chapter 3 tasks.'
    test_operators_defined(debug)

def test_task1(debug=False):
    test_repr(debug)
    test_repr_all_operators(debug)
    test_variables(debug)
    test_variables_all_operators(debug)
    test_operators(debug)
    test_operators_all_operators(debug)
    test_parse_prefix(debug)
    test_parse_prefix_all_operators(debug)
    test_is_formula(debug)
    test_is_formula_all_operators(debug)
    test_parse(debug)
    test_parse_all_operators(debug)

test_before_tasks(True)    
test_task1(True)
