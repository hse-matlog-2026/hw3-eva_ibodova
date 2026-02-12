from propositions.syntax_test import *
from propositions.semantics_test import *
from propositions.operators_test import *

def test_before_tasks(debug=False):
    assert is_binary('+'), 'Change is_binary() before testing Chapter 3 tasks.'
    test_operators_defined(debug)

def test_task3(debug=False):
    test_substitute_variables(debug)

test_before_tasks(True)    
test_task3(True)
