import itertools
import importlib
import os
from pathlib import Path

test_option = 1
output_folder = "reverse_cat_new"
output_filename = "reverse"

os.chdir(Path(__file__).parent.parent / "output" / output_folder)
z3 = importlib.import_module("output." + output_folder + "." + output_filename + "_Z3")
os.chdir(Path(__file__).parent.parent)
model = importlib.import_module("output." + output_folder + "." + output_filename)

if test_option == 1:

    nr_mismatches = 0
    for perm in itertools.permutations(['0', '1', '2', '3']):
        input = [z3.get_token_constant(x, z3.alphabet) for x in ['<s>'] + list(perm) + ['</s>']]
        predictions = z3.compute_original_predictions(input)
        model_predictions = model.run([str(x) for x in input])
        if([str(pred) for pred in predictions] != model_predictions):
            nr_mismatches += 1
            print("MISMATCH")
            print(f"Mismatch for input {input}")
            print(f"Z3 predictions: {predictions}")
            print(f"Model predictions: {model_predictions}")
    print("Total mismatches: ", nr_mismatches)