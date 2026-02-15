# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2025
# File name: propositions/semantics.py

"""Semantic analysis of propositional-logic constructs."""

from typing import AbstractSet, Iterable, Iterator, Mapping, Sequence, Tuple

from propositions.syntax import *
from propositions.proofs import *

#: A model for propositional-logic formulas, a mapping from variable names to
#: truth values.
Model = Mapping[str, bool]


def is_model(model: Model) -> bool:
    """Checks if the given dictionary is a model over some set of variable
    names.

    Parameters:
        model: dictionary to check.

    Returns:
        ``True`` if the given dictionary is a model over some set of variable
        names, ``False`` otherwise.
    """
    for key in model:
        if not is_variable(key):
            return False
    return True


def variables(model: Model) -> AbstractSet[str]:
    """Finds all variable names over which the given model is defined.

    Parameters:
        model: model to check.

    Returns:
        A set of all variable names over which the given model is defined.
    """
    assert is_model(model)
    return model.keys()


def evaluate(formula: Formula, model: Model) -> bool:
    """Calculates the truth value of the given formula in the given model.

    Parameters:
        formula: formula to calculate the truth value of.
        model: model over (possibly a superset of) the variable names of the
            given formula, to calculate the truth value in.

    Returns:
        The truth value of the given formula in the given model.

    Examples:
        >>> evaluate(Formula.parse('~(p&q76)'), {'p': True, 'q76': False})
        True

        >>> evaluate(Formula.parse('~(p&q76)'), {'p': True, 'q76': True})
        False
    """
    assert is_model(model)
    assert formula.variables().issubset(variables(model))

    root = formula.root

    if is_variable(root):
        return bool(model[root])

    if is_constant(root):
        return root == 'T'

    if is_unary(root):
        return not evaluate(formula.first, model)

    assert is_binary(root)
    if root == '&':
        return evaluate(formula.first, model) and evaluate(formula.second, model)

    if root == '|':
        return evaluate(formula.first, model) or evaluate(formula.second, model)

    if root == "->":
        return (not evaluate(formula.first, model)) or evaluate(formula.second, model)

    if root == "+":
        return evaluate(formula.first, model) != evaluate(formula.second, model)

    if root == "<->":
        return evaluate(formula.first, model) == evaluate(formula.second, model)

    if root == "-&":
        return not (evaluate(formula.first, model) and evaluate(formula.second, model))

    if root == "-|":
        return not (evaluate(formula.first, model) or evaluate(formula.second, model))
    
    assert False, f"unknown operator: {root}"


def all_models(variables: Sequence[str]) -> Iterable[Model]:
    """Calculates all possible models over the given variable names.

    Parameters:
        variables: variable names over which to calculate the models.

    Returns:
        An iterable over all possible models over the given variable names. The
        order of the models is lexicographic according to the order of the given
        variable names, where False precedes True.

    Examples:
        >>> list(all_models(['p', 'q']))
        [{'p': False, 'q': False}, {'p': False, 'q': True}, {'p': True, 'q': False}, {'p': True, 'q': True}]

        >>> list(all_models(['q', 'p']))
        [{'q': False, 'p': False}, {'q': False, 'p': True}, {'q': True, 'p': False}, {'q': True, 'p': True}]
    """
    for v in variables:
        assert is_variable(v)

    n = len(variables)

    for mask in range(2 ** n):
        m = {}
        for i, v in enumerate(variables):
            bit = (mask >> (n - 1 - i)) & 1
            m[v] = bool(bit)
        yield m


def truth_values(formula: Formula, models: Iterable[Model]) -> Iterable[bool]:
    """Calculates the truth value of the given formula in each of the given
    models.

    Parameters:
        formula: formula to calculate the truth value of.
        models: iterable over models to calculate the truth value in.

    Returns:
        An iterable over the respective truth values of the given formula in
        each of the given models, in the order of the given models.

    Examples:
        >>> list(truth_values(Formula.parse('~(p&q76)'), all_models(['p', 'q76'])))
        [True, True, True, False]
    """
    for m in models:
        yield evaluate(formula, m)


def print_truth_table(formula: Formula) -> None:
    """Prints the truth table of the given formula, with variable-name columns
    sorted alphabetically.

    Parameters:
        formula: formula to print the truth table of.

    Examples:
        >>> print_truth_table(Formula.parse('~(p&q76)'))
        | p | q76 | ~(p&q76) |
        |---|-----|----------|
        | F | F   | T        |
        | F | T   | T        |
        | T | F   | T        |
        | T | T   | F        |
    """
    vars_sorted = sorted(list(formula.variables()))
    header_cells = vars_sorted + [str(formula)]

    print('| ' + ' | '.join(header_cells) + ' |')

    print('|' + '|'.join('-' * (len(name) + 2) for name in header_cells) + '|')

    for m in all_models(vars_sorted):
        row = [('T' if m[v] else 'F') for v in vars_sorted]
        row.append('T' if evaluate(formula, m) else 'F')
        print('| ' + ' | '.join(row) + ' |')


def is_tautology(formula: Formula) -> bool:
    """Checks if the given formula is a tautology.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a tautology, ``False`` otherwise.
    """
    vars_sorted = sorted(list(formula.variables()))
    for m in all_models(vars_sorted):
        if not evaluate(formula, m):
            return False
    return True


def is_contradiction(formula: Formula) -> bool:
    """Checks if the given formula is a contradiction.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a contradiction, ``False`` otherwise.
    """
    vars_sorted = sorted(list(formula.variables()))
    for m in all_models(vars_sorted):
        if evaluate(formula, m):
            return False
    return True


def is_satisfiable(formula: Formula) -> bool:
    """Checks if the given formula is satisfiable.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is satisfiable, ``False`` otherwise.
    """
    vars_sorted = sorted(list(formula.variables()))
    for m in all_models(vars_sorted):
        if evaluate(formula, m):
            return True
    return False


def _synthesize_for_model(model: Model) -> Formula:
    """Synthesizes a propositional formula in the form of a single conjunctive
    clause that evaluates to ``True`` in the given model, and to ``False`` in
    any other model over the same variable names.

    Parameters:
        model: model over a nonempty set of variable names, in which the
            synthesized formula is to hold.

    Returns:
        The synthesized formula.
    """
    assert is_model(model)
    assert len(model.keys()) > 0

    vars_sorted = sorted(model.keys())

    v0 = vars_sorted[0]
    conj = Formula(v0) if model[v0] else Formula('~', Formula(v0))

    for v in vars_sorted[1:]:
        if model[v]:
            literal = Formula(v)
        else:
            literal = Formula('~', Formula(v))

        conj = Formula('&', conj, literal)

    return conj


def synthesize(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    """Synthesizes a propositional formula in DNF over the given variable names,
    that has the specified truth table.

    Parameters:
        variables: nonempty set of variable names for the synthesized formula.
        values: iterable over truth values for the synthesized formula in every
            possible model over the given variable names, in the order returned
            by `all_models`\\ ``(``\\ `~synthesize.variables`\\ ``)``.

    Returns:
        The synthesized formula.

    Examples:
        >>> formula = synthesize(['p', 'q'], [True, True, True, False])
        >>> for model in all_models(['p', 'q']):
        ...     evaluate(formula, model)
        True
        True
        True
        False
    """
    assert len(variables) > 0

    for v in variables:
        assert is_variable(v)

    models = list(all_models(list(variables)))
    values_list = list(values)
    assert len(values_list) == len(models)

    true_models = [models[i] for i, val in enumerate(values_list) if val]

    if len(true_models) == 0:
        p = variables[0]
        return Formula('&', Formula(p), Formula('~', Formula(p)))

    dnf = _synthesize_for_model(true_models[0])
    for m in true_models[1:]:
        dnf = Formula('|', dnf, _synthesize_for_model(m))

    return dnf


def _synthesize_for_all_except_model(model: Model) -> Formula:
    """Synthesizes a propositional formula in the form of a single disjunctive
    clause that evaluates to ``False`` in the given model, and to ``True`` in
    any other model over the same variable names.

    Parameters:
        model: model over a nonempty set of variable names, in which the
            synthesized formula is to not hold.

    Returns:
        The synthesized formula.
    """
    assert is_model(model)
    assert len(model.keys()) > 0

    vars_sorted = sorted(model.keys())

    v0 = vars_sorted[0]
    if model[v0]:
        disj = Formula('~', Formula(v0))
    else:
        disj = Formula(v0)

    for v in vars_sorted[1:]:
        if model[v]:
            clause = Formula('~', Formula(v))
        else:
            clause = Formula(v)

        disj = Formula('|', disj, clause)

    return disj


def synthesize_cnf(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    """Synthesizes a propositional formula in CNF over the given variable names,
    that has the specified truth table.

    Parameters:
        variables: nonempty set of variable names for the synthesized formula.
        values: iterable over truth values for the synthesized formula in every
            possible model over the given variable names, in the order returned
            by `all_models`\\ ``(``\\ `~synthesize.variables`\\ ``)``.

    Returns:
        The synthesized formula.

    Examples:
        >>> formula = synthesize_cnf(['p', 'q'], [True, True, True, False])
        >>> for model in all_models(['p', 'q']):
        ...     evaluate(formula, model)
        True
        True
        True
        False
    """
    assert len(variables) > 0
    for v in variables:
        assert is_variable(v)

    models = list(all_models(list(variables)))
    values_list = list(values)
    assert len(values_list) == len(models)

    false_models = [models[i] for i, val in enumerate(values_list) if not val]

    if len(false_models) == 0:
        p = variables[0]
        return Formula('|', Formula(p), Formula('~', Formula(p)))

    cnf = _synthesize_for_all_except_model(false_models[0])
    for m in false_models[1:]:
        cnf = Formula('&', cnf, _synthesize_for_all_except_model(m))
    return cnf


def evaluate_inference(rule: InferenceRule, model: Model) -> bool:
    """Checks if the given inference rule holds in the given model.

    Parameters:
        rule: inference rule to check.
        model: model to check in.

    Returns:
        ``True`` if the given inference rule holds in the given model, ``False``
        otherwise.

    Examples:
        >>> evaluate_inference(InferenceRule([Formula('p')], Formula('q')),
        ...                    {'p': True, 'q': False})
        False

        >>> evaluate_inference(InferenceRule([Formula('p')], Formula('q')),
        ...                    {'p': False, 'q': False})
        True
    """
    assert is_model(model)
    # Task 4.2


def is_sound_inference(rule: InferenceRule) -> bool:
    """Checks if the given inference rule is sound, i.e., whether its
    conclusion is a semantically correct implication of its assumptions.

    Parameters:
        rule: inference rule to check.

    Returns:
        ``True`` if the given inference rule is sound, ``False`` otherwise.
    """
    # Task 4.3
