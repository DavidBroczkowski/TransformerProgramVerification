"""
Here we are completely rewriting the logic of Z3 script generation so that it does not depend on intermediate Python code,
but directly extracts all the necessary information from the model itself (similar to how model_to_code works).
"""
from __future__ import annotations
import inspect
from pathlib import Path
from typing import Optional, Union, Sequence

from src.utils.code_utils import (
    get_unembed_df,
    get_embed_df,
    get_var_names,
    cat_head_to_code,
    num_head_to_code,
    cat_mlp_to_code,
    num_mlp_to_code,
    get_var_types
)
import torch
import pandas as pd


def _cat_head_to_z3(model, layer_idx: int, head_idx: int, idx_w: Sequence[str], autoregressive: bool = False) -> str:
    """
    Generates a Z3 predicate for a categorical attention head. An example result might look like a function (define-fun cat_head_{layer_idx}_{head_idx}) returning a Bool.
    Inside, you can describe a set of conditions (Or/And/If): for which (q, k) the head activates.
    It uses information about what the keys/queries are (like in cat_head_to_code, but without generating Python).
    """
    import torch
    import numpy as np

    attn = model.blocks[layer_idx].cat_attn
    W_K, W_Q, W_V = [W.detach().cpu() for W in (attn.W_K(), attn.W_Q(), attn.W_V())]
    W_pred = attn.W_pred.get_W().detach().cpu()
    pi_K, pi_Q, pi_V = [f.get_W().detach().cpu() for f in (attn.W_K, attn.W_Q, attn.W_V)]

    # Get variable names
    cat_var_names, _, _ = get_var_names(model, idx_w=idx_w)

    key_names = cat_var_names[pi_K.argmax(-1)]
    query_names = cat_var_names[pi_Q.argmax(-1)]
    val_names = cat_var_names[pi_V.argmax(-1)]

    if attn.W_K.n_heads == 1:
        key_names, query_names, val_names = [key_names], [query_names], [val_names]

    h = head_idx
    q, k, v = query_names[h], key_names[h], val_names[h]
    W_pred_h = W_pred[h]

    # Determine parameter types based on variable names
    q_is_position = "position" in str(q).lower()
    k_is_position = "position" in str(k).lower()

    q_type = "Int" if q_is_position else "String"
    k_type = "Int" if k_is_position else "String"

    q_name, k_name = f"{str(q)[:-1]}", f"{str(k)[:-1]}"
    if q_name == k_name:
        q_name, k_name = f"q_{q_name}", f"k_{k_name}"

    func_name = f"predicate_{layer_idx}_{head_idx}_expr"

    # Build conditions
    conditions = []
    for q_i in range(W_pred_h.shape[0]):
        k_j = (W_pred_h[q_i]).argmax(-1).item()

        # Format values based on type
        if q_is_position:
            q_val = f"IntVal({q_i})"
        else:
            if q_i < len(idx_w):
                q_val = f'StringVal("{idx_w[q_i]}")'
            else:
                q_val = f'StringVal("pad")'

        if k_is_position:
            k_val = f"IntVal({k_j})"
        else:
            if k_j < len(idx_w):
                k_val = f'StringVal("{idx_w[k_j]}")'
            else:
                k_val = f'StringVal("pad")'

        conditions.append(f"And({q_name} == {q_val}, {k_name} == {k_val})")

    # Combine conditions
    if len(conditions) == 0:
        z3_body = "False"
    elif len(conditions) == 1:
        z3_body = conditions[0]
    else:
        z3_body = "Or(" + ", ".join(conditions) + ")"

    return f"""
def {func_name}({q_name}, {k_name}):
    return {z3_body}
"""


def _num_head_to_z3(model, layer_idx: int, head_idx: int, autoregressive: bool = False) -> str:
    """
    Generates a Z3 predicate for a numerical attention head.  
    Similar to _cat_head_to_z3, but possibly compares numerical values directly  
    (for example, for q>k or q==k) – depending on the model's weights.
    """
    import torch
    import numpy as np

    attn = model.blocks[layer_idx].num_attn
    W_K, W_Q, W_V = [W.detach().cpu() for W in (attn.W_K(), attn.W_Q(), attn.W_V())]
    W_pred = attn.W_pred.get_W().detach().cpu()
    pi_K, pi_Q, pi_V = [f.get_W().detach().cpu() for f in (attn.W_K, attn.W_Q, attn.W_V)]

    # Get variable names
    cat_var_names, num_var_names, _ = get_var_names(model)

    key_names = cat_var_names[pi_K.argmax(-1)]
    query_names = cat_var_names[pi_Q.argmax(-1)]
    val_names = num_var_names[pi_V.argmax(-1)]

    if attn.W_K.n_heads == 1:
        key_names, query_names, val_names = [key_names], [query_names], [val_names]

    h = head_idx
    q, k, v = query_names[h], key_names[h], val_names[h]
    W_pred_h = W_pred[h]

    # Determine parameter types based on variable names
    q_is_position = "position" in str(q).lower()
    k_is_position = "position" in str(k).lower()

    q_type = "Int" if q_is_position else "String"
    k_type = "Int" if k_is_position else "String"

    q_name, k_name = f"{str(q)[:-1]}", f"{str(k)[:-1]}"
    if q_name == k_name:
        q_name, k_name = f"q_{q_name}", f"k_{k_name}"

    func_name = f"num_predicate_{layer_idx}_{head_idx}_expr"

    # Build conditions
    conditions = []
    for q_i in range(W_pred_h.shape[0]):
        k_j = (W_pred_h[q_i]).argmax(-1).item()

        # Format values based on type
        if q_is_position:
            q_val = f"IntVal({q_i})"
        else:
            q_val = f'StringVal("{q_i}")'

        if k_is_position:
            k_val = f"IntVal({k_j})"
        else:
            k_val = f'StringVal("{k_j}")'

        conditions.append(f"And({q_name} == {q_val}, {k_name} == {k_val})")

    # Combine conditions
    if len(conditions) == 0:
        z3_body = "False"
    elif len(conditions) == 1:
        z3_body = conditions[0]
    else:
        z3_body = "Or(" + ", ".join(conditions) + ")"

    return f"""
def {func_name}({q_name}, {k_name}):
    return {z3_body}
"""


def _cat_mlp_to_z3(model, layer_idx: int, mlp_idx: int, idx_w: Sequence[str]) -> str:
    """
    Generates an MLP expression for the categorical part.
    Suppose it is (define-fun cat_mlp_{layer_idx}_{mlp_idx} ((pos Int) (tok String)) Int).
    Returns some integer value, for example, varying depending on (pos, tok).
    """
    import torch
    import numpy as np
    import itertools

    mlp = model.blocks[layer_idx].cat_mlp.mlps[mlp_idx]
    n_vars = mlp.W_read.n_vars

    # Get variable names
    var_names, _, _ = get_var_names(model)

    mlp.eval()
    read = mlp.W_read
    with torch.no_grad():
        vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()

    mlp_vars_in, n_vars = read.W.shape
    var_dims = [mlp.d_out for _ in range(mlp_vars_in)]
    input_idxs = np.array(list(itertools.product(*[range(d) for d in var_dims])))

    X = np.zeros((len(input_idxs), read.d_in), dtype=np.float32)
    l = np.arange(X.shape[0])
    for i, j in enumerate(vars_in):
        X[l, input_idxs[:, i] + (var_dims[i] * j)] = 1

    X = torch.tensor(X, device=mlp.W_in.device)
    with torch.no_grad():
        mlp_out = mlp(X.unsqueeze(1)).squeeze(1).detach().cpu()

    mlp_var_out = mlp_out.argmax(-1).numpy()
    order = np.argsort(mlp_var_out)
    mlp_var_out = mlp_var_out[order]
    input_idxs = input_idxs[order]

    mlp_var_names = var_names[vars_in]

    # Determine parameter types based on variable names
    param_types = []
    for var in mlp_var_names:
        is_position = "position" in var.lower()
        param_types.append("Int" if is_position else "String")

    # Create parameter names
    param_names = [f"{v[:-1]}" for v in mlp_var_names]
    for i in range(len(param_names)):
        if param_names.count(param_names[i]) > 1:
            param_names[i] = f"param{i}_{param_names[i]}"

    func_name = f"mlp_{layer_idx}_{mlp_idx}_expr"

    # Build conditions
    conditions = []
    for i in range(len(input_idxs)):
        inputs = input_idxs[i]
        output = mlp_var_out[i]

        # Format condition
        cond_parts = []
        for j, (param_name, param_type, input_val) in enumerate(zip(param_names, param_types, inputs)):
            if param_type == "Int":
                val = f"IntVal({input_val})"
            else:
                if input_val < len(idx_w):
                    val = f'StringVal("{idx_w[input_val]}")'
                else:
                    val = f'StringVal("pad")'
            cond_parts.append(f"{param_name} == {val}")

        condition = "And(" + ", ".join(cond_parts) + ")"
        conditions.append((condition, output))

    # Group by output value
    output_to_conditions = {}
    for cond, out in conditions:
        if out not in output_to_conditions:
            output_to_conditions[out] = []
        output_to_conditions[out].append(cond)

    # Build the Z3 expression
    z3_body = []
    z3_body.append("    conds = [")
    for out_val, conds in output_to_conditions.items():
        for cond in conds:
            z3_body.append(f"        ({cond}, {out_val}),")
    z3_body.append("    ]")
    z3_body.append("")
    z3_body.append("    expr = IntVal(0)  # default value")
    z3_body.append("    for cond, val in reversed(conds):")
    z3_body.append("        expr = If(cond, val, expr)")
    z3_body.append("    return expr")

    param_list = ", ".join(param_names)

    return f"""
def {func_name}({param_list}):
{chr(10).join(z3_body)}
"""


def _num_mlp_to_z3(model, layer_idx: int, mlp_idx: int) -> str:
    """
    Generates an MLP expression for the numeric part.
    Suppose this is a similar function (define-fun num_mlp_{layer_idx}_{mlp_idx} ((pos Int) (val Int)) Int).
    """
    import torch
    import numpy as np
    import itertools

    mlp = model.blocks[layer_idx].num_mlp.mlps[mlp_idx]
    max_n = model.pos_embed.max_ctx * (layer_idx + 1)

    # Get variable names
    _, var_names, _ = get_var_names(model)

    mlp.eval()
    read = mlp.W_read
    with torch.no_grad():
        vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()

    mlp_vars_in, n_vars = read.W.shape
    var_dims = [max_n for _ in range(mlp_vars_in)]
    input_idxs = np.array(list(itertools.product(*[range(d) for d in var_dims])))

    # Limit the number of samples if there are too many
    if len(input_idxs) > 1000:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(input_idxs), 1000, replace=False)
        input_idxs = input_idxs[indices]

    X = np.zeros((len(input_idxs), read.d_in), dtype=np.float32)
    l = np.arange(X.shape[0])
    for i, j in enumerate(vars_in):
        X[l, j] = input_idxs[:, i]

    X = torch.tensor(X, device=mlp.W_in.device)
    with torch.no_grad():
        mlp_out = mlp(X.unsqueeze(1)).squeeze(1).detach().cpu()

    mlp_var_out = mlp_out.argmax(-1).numpy()
    mlp_var_names = var_names[vars_in]

    # Determine parameter types based on variable names
    param_types = []
    for var in mlp_var_names:
        is_position = "position" in var.lower()
        param_types.append("Int" if is_position else "String")

    # Create parameter names
    param_names = [f"{v[:-1]}" for v in mlp_var_names]
    for i in range(len(param_names)):
        if param_names.count(param_names[i]) > 1:
            param_names[i] = f"param{i}_{param_names[i]}"

    func_name = f"num_mlp_{layer_idx}_{mlp_idx}_expr"

    # Build conditions
    conditions = []
    for i in range(len(input_idxs)):
        inputs = input_idxs[i]
        output = mlp_var_out[i]

        # Format condition
        cond_parts = []
        for j, (param_name, param_type, input_val) in enumerate(zip(param_names, param_types, inputs)):
            if param_type == "Int":
                val = f"IntVal({input_val})"
            else:
                val = f'StringVal("{input_val}")'
            cond_parts.append(f"{param_name} == {val}")

        condition = "And(" + ", ".join(cond_parts) + ")"
        conditions.append((condition, output))

    # Group by output value
    output_to_conditions = {}
    for cond, out in conditions:
        if out not in output_to_conditions:
            output_to_conditions[out] = []
        output_to_conditions[out].append(cond)

    # Build the Z3 expression
    z3_body = []
    z3_body.append("    conds = [")
    for out_val, conds in output_to_conditions.items():
        for cond in conds:
            z3_body.append(f"        ({cond}, {out_val}),")
    z3_body.append("    ]")
    z3_body.append("")
    z3_body.append("    expr = IntVal(0)  # default value")
    z3_body.append("    for cond, val in reversed(conds):")
    z3_body.append("        expr = If(cond, val, expr)")
    z3_body.append("    return expr")

    param_list = ", ".join(param_names)

    return f"""
def {func_name}({param_list}):
{chr(10).join(z3_body)}
"""


def _generate_static_z3() -> str:
    """
    Generates static helper functions (select, aggregate, etc.).
    Returns a string with function definitions for a Z3 script.
    """
    lines = []

    # Headers
    lines.append("from z3 import *")
    lines.append("import pandas as pd")
    lines.append("import numpy as np")
    lines.append("")  # empty line

    # aggregate_expr
    lines.append("def aggregate_expr(attn_row, values):")
    lines.append("    # Takes from values[j] the j where attn_row[j] == True; if none - fallback to values[0]")
    lines.append("    expr = values[0]")
    lines.append("    # Iterate in reverse order to account for early indices in case of matches")
    lines.append("    for j in reversed(range(len(attn_row))):")
    lines.append("        expr = If(attn_row[j], values[j], expr)")
    lines.append("    return expr")
    lines.append("")

    # build_attention_block
    lines.append("def build_attention_block(solver, keys, queries, predicate_expr, values, name):")
    lines.append("    \"\"\"")
    lines.append("    Building an attention block:")
    lines.append("    - keys: list of elements (Int or String) for predicate_expr")
    lines.append("    - queries: list of elements (Int or String) to select from")
    lines.append("    - predicate_expr: function (q, k) -> BoolRef, defining the match condition")
    lines.append("    - values: list of elements (Int or String) for aggregate")
    lines.append("    - name: suffix for variable names in Z3")
    lines.append("    Returns a list of N outputs (String or Int), similar to `outs[...]`.")
    lines.append("    \"\"\"")
    lines.append("    N = len(keys)")
    lines.append("    # matrix of Bool variables attn[i][j]")
    lines.append("    attn = [[Bool(f\"attn_{name}_{i}_{j}\") for j in range(N)] for i in range(N)]")
    lines.append("    # flags: for each i, is there any match among keys[j]")
    lines.append("    any_match = [Bool(f\"any_{name}_{i}\") for i in range(N)]")
    lines.append("")
    lines.append("    # Calculate any_match[i] == Or(predicate_expr(queries[i], keys[j]) for j in range(N))")
    lines.append("    for i in range(N):")
    lines.append("        solver.add(any_match[i] == Or([predicate_expr(queries[i], keys[j]) for j in range(N)]))")
    lines.append("")
    lines.append("    # Determine output type: Int or String, depending on values")
    lines.append("    if values and isinstance(values[0], AstRef) and values[0].sort() == IntSort():")
    lines.append("        outputs = [Int(f\"attn_{name}_output_{i}\") for i in range(N)]")
    lines.append("    else:")
    lines.append("        outputs = [String(f\"attn_{name}_output_{i}\") for i in range(N)]")
    lines.append("")
    lines.append("    for i in range(N):")
    lines.append("        # exactly one True in row i")
    lines.append("        solver.add(Sum([If(attn[i][j], 1, 0) for j in range(N)]) == 1)")
    lines.append("")
    lines.append("        # for each j:")
    lines.append("        for j in range(N):")
    lines.append("            if j == 0:")
    lines.append("                # fallback: attn[i][0] can be True if there's no match, or if predicate_expr is true for (i,0)")
    lines.append("                solver.add(Implies(attn[i][0], Or(Not(any_match[i]), predicate_expr(queries[i], keys[0]))))")
    lines.append("            else:")
    lines.append("                # if attn[i][j] == True, then predicate_expr(queries[i], keys[j]) must be True")
    lines.append("                solver.add(Implies(attn[i][j], predicate_expr(queries[i], keys[j])))")
    lines.append("")
    lines.append("        # closest condition: if attn[i][k] and predicate_expr(queries[i], keys[j]) is true,")
    lines.append("        # then distance |i-k| <= |i-j| for all j.")
    lines.append("        for j in range(N):")
    lines.append("            for k in range(N):")
    lines.append("                solver.add(Implies(")
    lines.append("                    And(attn[i][k], predicate_expr(queries[i], keys[j])),")
    lines.append("                    Abs(i - k) <= Abs(i - j)")
    lines.append("                ))")
    lines.append("")
    lines.append("        # aggregate: select a value from values based on the attn[i] vector")
    lines.append("        solver.add(outputs[i] == aggregate_expr(attn[i], values))")
    lines.append("")
    lines.append("    return outputs")
    lines.append("")

    # build_mlp_block
    lines.append("def build_mlp_block(solver, positions, tokens, mlp_expr_fn, name):")
    lines.append("    \"\"\"")
    lines.append("    Building an MLP block: for each position i, create an Int output variable mlp_{name}_output_{i}")
    lines.append("    and constraint: output == mlp_expr_fn(position, token_at_position).")
    lines.append("    \"\"\"")
    lines.append("    N = len(tokens)")
    lines.append("    outputs = [Int(f\"mlp_{name}_output_{i}\") for i in range(N)]")
    lines.append("    for i in range(N):")
    lines.append("        solver.add(outputs[i] == mlp_expr_fn(positions[i], tokens[i]))")
    lines.append("    return outputs")
    lines.append("")

    return "\n".join(lines)


def _generate_build_pipeline_by_run(model) -> str:
    """
    Returns Z3 functions/expressions that reflect the logic of run(...).
    Automatically determines the order of layers and their outputs based on the model's structure.
    """
    lines = []
    lines.append("def build_pipeline(solver, tokens, position_vars):")
    lines.append('    """')
    lines.append("    Pulls all attention-, MLP-blocks, generates logits and pred[i] variables.")
    lines.append("    Returns dictionaries outputs_by_name, logits, pred_vars.")
    lines.append('    """')
    lines.append("    N = len(tokens)")
    lines.append("    # === Attention + MLP ===")
    lines.append("    outs = {}")

    def map_var(var_name):
        """Map variable names to Z3 variables"""
        if var_name == "tokens":
            return "tokens"
        if var_name == "positions":
            return "position_vars"
        if var_name.endswith("_outputs"):
            layer = var_name[:-len("_outputs")]
            return f'outs["{layer}"]'
        # For attention heads, map to previous outputs if available
        if var_name.startswith("attn_") and "_" in var_name:
            parts = var_name.split("_")
            if len(parts) >= 3:
                layer = parts[1]
                head = parts[2]
                return f'outs["attn_{layer}_{head}"]'
        # For MLP outputs
        if var_name.startswith("mlp_") and "_" in var_name:
            parts = var_name.split("_")
            if len(parts) >= 3:
                layer = parts[1]
                mlp = parts[2]
                return f'outs["mlp_{layer}_{mlp}"]'
        # Default fallback
        return var_name

    # Generate blocks in order based on model structure
    for layer_idx, block in enumerate(model.blocks):
        # Categorical attention heads
        for head_idx in range(block.n_heads_cat):
            attn = block.cat_attn
            pi_K, pi_Q, pi_V = [f.get_W().detach().cpu() for f in (attn.W_K, attn.W_Q, attn.W_V)]

            # Get variable names
            cat_var_names, _, _ = get_var_names(model)

            key_names = cat_var_names[pi_K.argmax(-1)]
            query_names = cat_var_names[pi_Q.argmax(-1)]
            val_names = cat_var_names[pi_V.argmax(-1)]

            if attn.W_K.n_heads == 1:
                key_names, query_names, val_names = [key_names], [query_names], [val_names]

            k, q, v = key_names[head_idx], query_names[head_idx], val_names[head_idx]

            # Map variable names to Z3 variables
            keys_mapped = map_var(str(k))
            queries_mapped = map_var(str(q))
            values_mapped = map_var(str(v))

            lines.append(
                f'    outs["attn_{layer_idx}_{head_idx}"] = build_attention_block(solver, '
                f'{keys_mapped}, {queries_mapped}, predicate_{layer_idx}_{head_idx}_expr, '
                f'{values_mapped}, "{layer_idx}_{head_idx}")'
            )

        # Numerical attention heads
        for head_idx in range(block.n_heads_num):
            attn = block.num_attn
            pi_K, pi_Q, pi_V = [f.get_W().detach().cpu() for f in (attn.W_K, attn.W_Q, attn.W_V)]

            # Get variable names
            cat_var_names, num_var_names, _ = get_var_names(model)

            key_names = cat_var_names[pi_K.argmax(-1)]
            query_names = cat_var_names[pi_Q.argmax(-1)]
            val_names = num_var_names[pi_V.argmax(-1)]

            if attn.W_K.n_heads == 1:
                key_names, query_names, val_names = [key_names], [query_names], [val_names]

            k, q, v = key_names[head_idx], query_names[head_idx], val_names[head_idx]

            # Map variable names to Z3 variables
            keys_mapped = map_var(str(k))
            queries_mapped = map_var(str(q))
            values_mapped = "[IntVal(1)] * N"  # Numerical values are typically 1s

            lines.append(
                f'    outs["num_attn_{layer_idx}_{head_idx}"] = build_attention_block(solver, '
                f'{keys_mapped}, {queries_mapped}, num_predicate_{layer_idx}_{head_idx}_expr, '
                f'{values_mapped}, "num_{layer_idx}_{head_idx}")'
            )

        # Categorical MLPs
        for mlp_idx in range(block.n_cat_mlps):
            mlp = block.cat_mlp.mlps[mlp_idx]
            read = mlp.W_read
            with torch.no_grad():
                vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()

            # Get variable names
            var_names, _, _ = get_var_names(model)
            mlp_var_names = var_names[vars_in]

            # Map variable names to Z3 variables
            pos_mapped = map_var(str(mlp_var_names[0])) if len(mlp_var_names) > 0 else "position_vars"
            input_mapped = map_var(str(mlp_var_names[1])) if len(mlp_var_names) > 1 else "tokens"

            lines.append(
                f'    outs["mlp_{layer_idx}_{mlp_idx}"] = build_mlp_block(solver, '
                f'{pos_mapped}, {input_mapped}, mlp_{layer_idx}_{mlp_idx}_expr, "{layer_idx}_{mlp_idx}")'
            )

        # Numerical MLPs
        for mlp_idx in range(block.n_num_mlps):
            mlp = block.num_mlp.mlps[mlp_idx]
            read = mlp.W_read
            with torch.no_grad():
                vars_in = torch.argmax(read.W, dim=-1).cpu().numpy()

            # Get variable names
            _, var_names, _ = get_var_names(model)
            mlp_var_names = var_names[vars_in]

            # Map variable names to Z3 variables
            pos_mapped = map_var(str(mlp_var_names[0])) if len(mlp_var_names) > 0 else "position_vars"
            input_mapped = map_var(str(mlp_var_names[1])) if len(mlp_var_names) > 1 else "tokens"

            lines.append(
                f'    outs["num_mlp_{layer_idx}_{mlp_idx}"] = build_mlp_block(solver, '
                f'{pos_mapped}, {input_mapped}, num_mlp_{layer_idx}_{mlp_idx}_expr, "num_{layer_idx}_{mlp_idx}")'
            )

    # === Logits ===
    lines.append("    # === Logits ===")
    lines.append("    logits = {(i, cls): Real(f\"logit_{i}_{cls}\") for i in range(N) for cls in classes}")
    lines.append("    features = {")
    lines.append('        "tokens": tokens,')
    lines.append('        "positions": position_vars,')
    lines.append('        "ones": [IntVal(1)] * N,')

    # Add all outputs to features
    for layer_idx, block in enumerate(model.blocks):
        for head_idx in range(block.n_heads_cat):
            lines.append(f'        "attn_{layer_idx}_{head_idx}_outputs": outs["attn_{layer_idx}_{head_idx}"],')
        for head_idx in range(block.n_heads_num):
            lines.append(f'        "num_attn_{layer_idx}_{head_idx}_outputs": outs["num_attn_{layer_idx}_{head_idx}"],')
        for mlp_idx in range(block.n_cat_mlps):
            lines.append(f'        "mlp_{layer_idx}_{mlp_idx}_outputs": outs["mlp_{layer_idx}_{mlp_idx}"],')
        for mlp_idx in range(block.n_num_mlps):
            lines.append(f'        "num_mlp_{layer_idx}_{mlp_idx}_outputs": outs["num_mlp_{layer_idx}_{mlp_idx}"],')

    lines.append("    }")

    # Equations for logits
    lines.append("    # for each (i,cls) one equation logit = Sum(If(...))")
    lines.append("    for i in range(N):")
    lines.append("        for cls in classes:")
    lines.append("            contribs = []")
    lines.append("            for feat_name, exprs in features.items():")
    lines.append("                feat_var = exprs[i]")
    lines.append("                for ((f_name, f_val), weights) in classifier_weights.iterrows():")
    lines.append("                    if f_name != feat_name:")
    lines.append("                        continue")
    lines.append("                    w = RealVal(str(weights[cls]))")
    lines.append("                    if feat_name == 'ones':")
    lines.append("                        contribs.append(w)")
    lines.append("                    else:")
    lines.append("                        if isinstance(feat_var, AstRef) and feat_var.sort() == StringSort():")
    lines.append("                            const = StringVal(f_val)")
    lines.append("                        else:")
    lines.append("                            const = IntVal(int(f_val))")
    lines.append("                        contribs.append(If(feat_var == const, w, RealVal('0')))")
    lines.append("            solver.add(logits[(i, cls)] == Sum(contribs))")

    # === Predictions ===
    lines.append("")
    lines.append("    # === Predictions ===")
    lines.append("    pred = [String(f\"pred_{i}\") for i in range(N)]")
    lines.append("    for i in range(N):")
    lines.append("        if i == 0:")
    lines.append("            solver.add(pred[i] == tokens[0])")
    lines.append("        elif i == N-1:")
    lines.append("            solver.add(pred[i] == tokens[N-1])")
    lines.append("        else:")
    lines.append("            for cls in classes:")
    lines.append("                cond = And([logits[(i, cls)] >= logits[(i, o)] for o in classes if o != cls])")
    lines.append("                solver.add(Implies(cond, pred[i] == StringVal(cls)))")
    lines.append("")
    lines.append("    return outs, logits, pred")

    return "\n".join(lines)


def _generate_predictions_code() -> str:
    """
    Generates Z3 code that computes the 'original' model predictions.
    Creates a function compute_original_predictions, which calls build_pipeline and then extracts the predictions.
    """
    return """
def compute_original_predictions(input_tokens):
    N = len(input_tokens)
    s1 = Solver()
    # 1. Variables and fixing input_tokens
    tokens = [String(f"token_{i}") for i in range(N)]
    for i, val in enumerate(input_tokens):
        s1.add(tokens[i] == StringVal(val))
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s1.add(pos[i] == IntVal(i))
    # 2. Run the pipeline
    _, logits, pred_orig_vars = build_pipeline(s1, tokens, pos)
    assert s1.check() == sat
    m = s1.model()

    # 3. Extract concrete strings
    return [str(m.evaluate(pred_orig_vars[i]).as_string()) for i in range(N)]
"""


def model_to_Z3(
    model,
    idx_w: Sequence[str],
    idx_t: Sequence[str],
    *,
    embed_csv: bool = False,
    embed_enums: bool = False,
    unembed_csv: bool = True,
    one_hot: bool = False,
    autoregressive: bool = False,
    var_types=None,
    output_dir: Union[str, Path] = ".",
    name: str = "program",
    save: bool = True,
) -> str:
    """
    Generates a ready-made Z3 script directly from the model, without an intermediate Python step.
    The logic is similar to what model_to_code does: we simply go through the layer/heads/MLP and assemble it into Z3 format.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If necessary, we collect weights_df and embed_df here
    weights_path: Optional[Path] = None
    embed_path: Optional[Path] = None

    if var_types == True:
        var_types = get_var_types(
            model, idx_w, one_hot=one_hot, enums=embed_enums
        )

    if unembed_csv:
        # Generate classifier weights CSV
        weights_df = get_unembed_df(model, idx_t, var_types=var_types, one_hot=one_hot, enums=embed_enums)
        weights_path = output_dir / f"{name}_weights.csv"
        if save:
            print(f"Writing classifier weights to {weights_path.as_posix()}")
            weights_df.to_csv(weights_path.as_posix())

    if embed_csv:
        # Generate embeddings CSV
        embed_df = get_embed_df(model.embed, idx_w)
        embed_path = output_dir / f"{name}_embeddings.csv"
        if save:
            print(f"Writing embeddings to {embed_path.as_posix()}")
            embed_df.to_csv(embed_path.as_posix())

    # Generating static functions
    static_code = _generate_static_z3()

    # Generating attention predicates
    predicate_blocks = []
    for layer_idx, block in enumerate(model.blocks):
        for head_idx in range(block.n_heads_cat):
            predicate_blocks.append(_cat_head_to_z3(model, layer_idx, head_idx, idx_w, autoregressive))

        for head_idx in range(block.n_heads_num):
            predicate_blocks.append(_num_head_to_z3(model, layer_idx, head_idx, autoregressive))

    predicates_code = "\n".join(predicate_blocks)

    # Generating MLP expressions
    mlp_blocks = []
    for layer_idx, block in enumerate(model.blocks):
        for mlp_idx in range(block.n_cat_mlps):
            mlp_blocks.append(_cat_mlp_to_z3(model, layer_idx, mlp_idx, idx_w))

        for mlp_idx in range(block.n_num_mlps):
            mlp_blocks.append(_num_mlp_to_z3(model, layer_idx, mlp_idx))

    mlp_code = "\n".join(mlp_blocks)

    # Generating build_pipeline (analog of run)
    build_pipeline_code = _generate_build_pipeline_by_run(model)

    # Generating prediction computation
    compute_pred_code = _generate_predictions_code()

    # Generate weight reading code
    weight_reading_code = []
    if weights_path:
        weight_reading_code.append("# —————— Read weights and set up constants ——————")
        weight_reading_code.append(f'classifier_weights = pd.read_csv("{weights_path.name}", index_col=[0, 1], dtype={{"feature": str}})')
        weight_reading_code.append("classes = classifier_weights.columns.tolist()")
        weight_reading_code.append("")

    script_parts = [
        "# ================================================",
        "#  Auto-generated Z3 model script",
        "# ================================================",
        "",
        "# --- Static helper functions ---",
        static_code,
        "",
    ]

    if weight_reading_code:
        script_parts.extend(weight_reading_code)

    script_parts.extend([
        "# --- Attention predicates ---",
        predicates_code,
        "",
        "# --- MLP expressions ---",
        mlp_code,
        "",
        "# --- Pipeline builder ---",
        build_pipeline_code,
        "",
        "# --- Original model predictions (reference) ---",
        compute_pred_code,
        "",
    ])

    # Add example usage
    script_parts.extend([
        "# --- Example usage ---",
        "if __name__ == '__main__':",
        "    # Example input",
        f"    example_input = {list(idx_w[:5]) + ['</s>']}",
        "    predictions = compute_original_predictions(example_input)",
        "    print(f\"Input: {example_input}\")",
        "    print(f\"Predictions: {predictions}\")",
        ""
    ])

    full_script = "\n".join(script_parts)

    # (Optional) Save to disk
    if save:
        out_file = output_dir / f"{name}_Z3.py"
        out_file.write_text(full_script, encoding="utf-8")
        print(f"[z3_script_generator] Z3 script written to: {out_file.resolve()}")

    return full_script


def export_model_to_Z3(
    model,
    idx_w,
    idx_t,
    *,
    output_dir: Union[str, Path] = ".",
    name: str = "program",
    **kwargs,
) -> Path:
    """
    Standard wrapper: takes a Z3 script and immediately writes it to a file.
    """
    output_dir = Path(output_dir).as_posix()
    output_file = output_dir / f"{name}_Z3.py"
    script = model_to_Z3(model, idx_w, idx_t, output_dir=output_dir, name=name, save=True, **kwargs)
    return output_file
