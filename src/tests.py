import itertools
import importlib
import os
import math
from pathlib import Path
from tqdm import tqdm
from z3 import *

test_option = 3  # 1: test all 4-permutations; 2: verify property
output_folder = "sort"
output_filename = "sort"

os.chdir(Path(__file__).parent.parent / "output" / output_folder)
solver = importlib.import_module("output." + output_folder + "." + output_filename + "_Z3")
os.chdir(Path(__file__).parent.parent)
model = importlib.import_module("output." + output_folder + "." + output_filename)

def build_solver(input_length, enforce_start_end):
    N = input_length
    s = Solver()

    tokens = [Const(f"token_{i}", solver.Token) for i in range(N)]
    
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s.add(pos[i] == IntVal(i))

    outs, logits, predictions = solver.build_pipeline(s, tokens, pos, N > 0 and enforce_start_end)

    return (s, tokens, pos, outs, logits, predictions)

def compute_original_predictions(input_tokens, build_solver_out = None):

    N = len(input_tokens)
    
    if build_solver_out is None:
        s = Solver()

        tokens = [Const(f"token_{i}", solver.Token) for i in range(N)]
        for i, val in enumerate(input_tokens):
            s.add(tokens[i] == val)
        
        pos = [Int(f"pos_{i}") for i in range(N)]
        for i in range(N):
            s.add(pos[i] == IntVal(i))
        
        outs, logits, predictions = solver.build_pipeline(s, tokens, pos, N > 0 and input_tokens[0] == solver.Token_start)
    
    else:
        (s, tokens, pos, outs, logits, predictions) = build_solver_out
        s = s.__copy__()
        for i, val in enumerate(input_tokens):
            s.add(tokens[i] == val)

    assert s.check() == sat
    m = s.model()

    return [m.evaluate(predictions[i]) for i in range(N)], outs, m

def find_adversarial_sorting(input_tokens, vocab):
    N = len(input_tokens)
    #pred_orig, _, _ = compute_original_predictions(input_tokens)
    pred_orig, _ = model.run([str(x) for x in input_tokens])
    pred_orig = [solver.get_enum_constant(x, solver.class_name_to_val) for x in pred_orig]
    print("Original preds:", pred_orig)

    s2 = Solver()
    # 1. New adversarial token variables
    tokens_adv = [Const(f"tok_adv_{i}", solver.Token) for i in range(N)]
    s2.add(tokens_adv[0] == solver.Token_start)
    s2.add(tokens_adv[-1] == solver.Token_end)

    for tok in tokens_adv:
        # Only valid tokens
        s2.add(Or([tok == v for v in input_tokens]))

    orig_inner = input_tokens[1:-1]
    adv_inner = tokens_adv[1:-1]

    s2.add(is_permutation_z3(orig_inner, adv_inner, vocab))

    # 2. Positions (the same)
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s2.add(pos[i] == IntVal(i))

    # 3. Run pipeline on adversarial
    _, logits, pred_adv_vars = solver.build_pipeline(s2, tokens_adv, pos)

    # 4. Prediction difference
    s2.add(Or([pred_adv_vars[i] != pred_orig[i] for i in range(N)]))

    if s2.check() == sat:
        m2 = s2.model()
        adv = [m2.evaluate(tokens_adv[i]) for i in range(N)]
        print("Adversarial example:", adv)
        print("New preds:", [m2.evaluate(pred_adv_vars[i]) for i in range(N)])
    else:
        print(f"No adversarial example found.")

def is_permutation_z3(predictions, input, vocab):
    """
    Returns Z3-condition that adv_tokens is a permutation of orig_tokens
    for tokens from the given vocabulary vocab (for example, ["0", "1", ..., "4"])
    """
    constraints = []
    for val in vocab:
        if predictions[0].sort() != solver.Token:
            count_adv = Sum([If(solver.token_equals_class(val, tok), 1, 0) for tok in predictions])
        else:
            count_adv = Sum([If(tok == val, 1, 0) for tok in predictions])
        count_orig = Sum([If(tok == val, 1, 0) for tok in input])
        constraints.append(count_adv == count_orig)
    return And(constraints)

def is_sorted(predictions):
    return And([
        class_to_int(predictions[i]) <= class_to_int(predictions[i + 1]) for i in range(1, len(predictions) - 2)
    ])

def sorted_property(vocab, input, predictions):
    return And(
        is_sorted(predictions[1:-1]),
        is_permutation_z3(predictions[1:-1], input[1:-1], vocab)
    ) 

def reverse_property(vocab, input, predictions):
    N = len(input)
    return And([
        solver.token_equals_class(input[N - 1 - i], predictions[i]) for i in range(1, N - 1)
    ])

def check_property(vocab, N, enforce_start_end, property_func, solver_data=None):
   
    if solver_data is None:
        s = Solver()

        # Variables: arbitrary tokens
        tokens = [Const(f"tok_{i}", solver.Token) for i in range(N)]
        pos = [IntVal(i) for i in range(N)]

        # Add pipeline and get predictions
        solver_outs, _, pred_vars = solver.build_pipeline(s, tokens, pos)
    
    else:
        (s, tokens, pos, solver_outs, _, pred_vars) = solver_data
        s = s.__copy__()

    # Restrict to valid tokens from the vocabulary
    for i, tok in enumerate(tokens):
        if i == 0 and enforce_start_end:
            s.add(tok == solver.Token_start)
        elif i == N - 1 and enforce_start_end:
            s.add(tok == solver.Token_end)
        else:
            s.add(Or([tok == v for v in vocab]))

    # Prediction should be NOT satisfying the property
    s.add(Not(property_func(vocab, tokens, pred_vars)))

    # Property verification
    if s.check() == sat:
        m = s.model()
        input_example = [m.evaluate(tok) for tok in tokens]
        output_example = [m.evaluate(pred) for pred in pred_vars]
        model_output, model_outs = model.run([str(x) for x in input_example])
        print("Counterexample found!")
        print("Input:", input_example)
        print("Output:", output_example)
        print("Actual Model output:", model_output)
        if([str(x) for x in output_example] != model_output):
            print_variables(solver_outs, model_outs, m)
        return False
    else:
        print("The transformer program fulfills the property on all possible inputs.")
        return True

def print_variables(solver_outs, model_outs, solver_model, print_interleaved=True):
    if not print_interleaved:
        print("Z3 outputs:")
        for varname, vars in solver_outs.items():
            print(f"{varname} = {[solver_model.evaluate(var) for var in vars]}")
        print("Model outputs:")
        for vars in model_outs:
            print(f"{vars[0]} = {vars[1]}")
    else:
        print("Interleaved outputs:")
        for varname, vars in solver_outs.items():
            model_out = list(filter(lambda x: x[0] == (varname + "_outputs"), model_outs))
            if(len(model_out) > 0):
                model_out = model_out[0]
                print(f"Z3    {varname} = {[solver_model.evaluate(var) for var in vars]}")
                print(f"Model {varname} = {model_out[1]}")

token_to_int = Function('token_to_int', solver.Token, IntSort())
token_to_int_constraints = [token_to_int(nr) == IntVal(int(str(nr))) for nr in solver.alphabet if str(nr) not in ['<s>', '</s>', '<pad>']]
class_to_int = Function('class_to_int', solver.Class, IntSort())
class_to_int_constraints = [class_to_int(nr) == IntVal(int(str(nr))) for nr in solver.classes_constants if str(nr) not in ['<s>', '</s>', '<pad>']]

if test_option == 1:

    nr_mismatches = 0
    items = ['0', '1', '2', '3']
    perms = itertools.permutations(items)
    total = math.factorial(len(items))
    solver_data = build_solver(len(items) + 2, True)
    for perm in tqdm(perms, total=total, desc="testing perms"):
        input = [solver.get_enum_constant(x, solver.token_name_to_val) for x in ['<s>'] + list(perm) + ['</s>']]
        predictions, solver_outs, solver_model = compute_original_predictions(input, solver_data)
        model_predictions, model_outs = model.run([str(x) for x in input])
        if([str(pred) for pred in predictions] != model_predictions):
            nr_mismatches += 1
            print("--------MISMATCH--------")
            print(f"Mismatch for input {input}")
            print(f"Z3 predictions: {predictions}")
            print(f"Model predictions: {model_predictions}")
            print_variables(solver_outs, model_outs, solver_model)
    print("Total mismatches: ", nr_mismatches)

if test_option == 2:

    for N in range(3, 9):
        print(f"Verifying property for input length N={N}...")
        solver_data = build_solver(N, True)
        solver_data[0].add(class_to_int_constraints)
        print("Solve...")
        result = check_property(
            vocab=[tok for tok in solver.alphabet if str(tok) not in ['<s>', '</s>', '<pad>']],
            N=N,
            enforce_start_end=True,
            property_func=sorted_property,
            solver_data=solver_data
        )
        if not result:
            break

if test_option == 3:
    input_tokens = [solver.get_enum_constant(x, solver.token_name_to_val) for x in ['<s>', '1', '0', '0', '0', '0', '</s>']]
    print("Finding adversarial example for input:", input_tokens)
    find_adversarial_sorting(input_tokens, [tok for tok in solver.alphabet if str(tok) not in ['<s>', '</s>', '<pad>']])