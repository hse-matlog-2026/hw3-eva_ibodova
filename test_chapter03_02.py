from propositions.syntax_test import *
from propositions.semantics_test import *
from propositions.operators_test import *

def test_before_tasks(debug=False):
    assert is_binary('+'), 'Change is_binary() before testing Chapter 3 tasks.'
    test_operators_defined(debug)
         
def test_task2(debug=False):
    test_evaluate(debug)
    test_evaluate_all_operators(debug)
    test_truth_values(debug)
    test_is_tautology(debug)
    test_is_contradiction(debug)
    test_is_satisfiable(debug)
    test_is_tautology_all_operators(debug)
    test_print_truth_table(debug)

test_before_tasks(True)    
test_task2(True)
