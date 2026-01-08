import itertools
import importlib
import os
import math
from pathlib import Path
from tqdm import tqdm
from z3 import *

test_option = 1
output_folder = "reverse_num_new"
output_filename = "reverse"

os.chdir(Path(__file__).parent.parent / "output" / output_folder)
solver = importlib.import_module("output." + output_folder + "." + output_filename + "_Z3")
os.chdir(Path(__file__).parent.parent)
model = importlib.import_module("output." + output_folder + "." + output_filename)

def compute_original_predictions(input_tokens):
    N = len(input_tokens)
    s1 = Solver()

    # 1. Variables and fixing input_tokens
    
    tokens = [Const(f"token_{i}", solver.Token) for i in range(N)]
    for i, val in enumerate(input_tokens):
        s1.add(tokens[i] == val)
    
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s1.add(pos[i] == IntVal(i))
    
    # 2. Run the pipeline
    outs, logits, pred_orig_vars = solver.build_pipeline(s1, tokens, pos, input_tokens)
    #print(s1.check() == sat)
    assert s1.check() == sat
    m = s1.model()

    # 3. Extract concrete strings
    return [m.evaluate(pred_orig_vars[i]) for i in range(N)], outs, m

if test_option == 1:

    nr_mismatches = 0
    items = ['0', '1', '2', '3']
    perms = itertools.permutations(items)
    total = math.factorial(len(items))
    for perm in tqdm(perms, total=total, desc="testing perms"):
        input = [solver.get_token_constant(x, solver.alphabet) for x in ['<s>'] + list(perm) + ['</s>']]
        predictions, solver_outs, solver_model = compute_original_predictions(input)
        model_predictions, model_outs = model.run([str(x) for x in input])
        if([str(pred) for pred in predictions] != model_predictions):
            nr_mismatches += 1
            print("--------MISMATCH--------")
            print(f"Mismatch for input {input}")
            print(f"Z3 predictions: {predictions}")
            print(f"Model predictions: {model_predictions}")
            print("Z3 outputs:")
            for varname, vars in solver_outs.items():
                print(f"{varname} = {[solver_model.evaluate(var) for var in vars]}")
            print("Model outputs:")
            for vars in model_outs:
                print(f"{vars[0]} = {vars[1]}")
    print("Total mismatches: ", nr_mismatches)

