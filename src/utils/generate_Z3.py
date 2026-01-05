import os
import ast

def generate_static_z3_code():
    """
    Returns a string with static Z3 code blocks:
    - imports
    - helper functions: aggregate_expr, build_attention_block, build_mlp_block, etc.
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


def extract_predicate_functions(source_code):
    """
    Parses source_code and returns a list of ast.FunctionDef for functions
    whose names start with "predicate_", regardless of nesting level
    (including inside the run function).
    """
    module = ast.parse(source_code)
    preds = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("predicate_"):
            preds.append(node)
    return preds


def parse_predicate_function(func_def: ast.FunctionDef):
    """
    Parses an ast.FunctionDef whose body is a chain of if/elif statements,
    each returning a condition of the form `other_arg == literal`.
    Returns a list of tuples:
        [(set_of_values_for_param1, literal_for_param2, param1_name, param2_name), ...]
    and, possibly, default_return (literal or None).

    We assume the definition looks something like this:
        def predicate_X(arg1, arg2):
            if arg1 in {a, b, ...}:
                return arg2 == L1
            elif arg1 in {...}:
                return arg2 == L2
            ...
            # (possibly) else:
            #     return arg2 == Ldefault
    We return:
      clauses = [
        ( [a, b, ...], L1 ),
        ( [...], L2 ),
        ...
      ]
    If there's an else: return ..., we add it as the last element with a key of None for the set or a special key.
    """
    clauses = []
    default_clause = None  # if there's an explicit else
    # parameter names
    if len(func_def.args.args) != 2:
        # in this simplified version we expect exactly 2 arguments
        return None
    param1 = func_def.args.args[0].arg
    param2 = func_def.args.args[1].arg

    # helper: process one ast.If and its orelse branch recursively
    def process_if(if_node):
        nonlocal default_clause
        # Parse the if_node.test condition, then the if_node.body, then orelse
        # We expect the body to contain exactly one Return: return <expr>
        # test: either Compare with In (membership), or comparison == (but in predicate usually membership on first arg)
        test = if_node.test
        # get the set of values for param1
        vals = []
        # test can be:
        #  - Compare(left=Name(param1), ops=[In], comparators=[Set(elts=...)]), or
        #  - Could be Or(...) compound, but in our simple pattern: if position in {..}
        if isinstance(test, ast.Compare):
            # membership variant: left Name(param1), ops [In], comparator Set
            if (isinstance(test.left, ast.Name) and test.left.id == param1
                    and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
                    and len(test.comparators) == 1 and isinstance(test.comparators[0], (ast.Set, ast.List, ast.Tuple))):
                # collect values
                for elt in test.comparators[0].elts:
                    if isinstance(elt, ast.Constant):
                        vals.append(elt.value)
                    else:
                        # more complex case (e.g., Name), skip
                        pass
            else:
                # in a simple predicate, they might not use membership but direct comparison param1 == literal?
                # for example: if param1 == 3: ...
                if (isinstance(test.left, ast.Name) and test.left.id == param1
                        and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                        and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)):
                    vals.append(test.comparators[0].value)
                else:
                    # unclear test: return None or ignore
                    # We could extend the logic, but for now - only membership and direct equality
                    return
        else:
            # condition is not Compare: could be a complex Or(...) inside test?
            # we could add processing for ast.BoolOp with And/Or, but for now we'll skip
            return

        # Parse Return in the body
        # we expect that body has exactly one ast.Return
        ret_nodes = [n for n in if_node.body if isinstance(n, ast.Return)]
        if len(ret_nodes) != 1:
            # if there are none or more than one, skip
            return
        ret = ret_nodes[0].value
        # Expect Compare: Name(param2) == Constant or vice versa
        literal = None
        # ret can be Compare:
        if isinstance(ret, ast.Compare):
            # left param2?
            if isinstance(ret.left, ast.Name) and ret.left.id == param2:
                # comparison param2 == literal?
                if len(ret.ops) == 1 and isinstance(ret.ops[0], ast.Eq) and len(ret.comparators) == 1 and isinstance(
                        ret.comparators[0], ast.Constant):
                    literal = ret.comparators[0].value
            # or literal == param2?
            elif isinstance(ret.comparators[0], ast.Name) and ret.comparators[0].id == param2:
                if len(ret.ops) == 1 and isinstance(ret.ops[0], ast.Eq) and isinstance(ret.left, ast.Constant):
                    literal = ret.left.value
        # If not recognized, skip
        if literal is None:
            return

        # Add to clauses
        clauses.append((vals, literal))

        # Now process orelse
        if if_node.orelse:
            # orelse can be either [If(...)] for elif, or [Return(...)] for else, or a combination
            if len(if_node.orelse) == 1 and isinstance(if_node.orelse[0], ast.If):
                process_if(if_node.orelse[0])
            else:
                # Check if there's a Return in orelse: default
                for stmt in if_node.orelse:
                    if isinstance(stmt, ast.Return):
                        # Similarly, parse Return
                        ret2 = stmt.value
                        literal2 = None
                        if isinstance(ret2, ast.Compare):
                            # similar to previous logic
                            if isinstance(ret2.left, ast.Name) and ret2.left.id == param2:
                                if len(ret2.ops) == 1 and isinstance(ret2.ops[0], ast.Eq) and len(
                                        ret2.comparators) == 1 and isinstance(ret2.comparators[0], ast.Constant):
                                    literal2 = ret2.comparators[0].value
                            elif len(ret2.comparators) == 1 and isinstance(ret2.comparators[0], ast.Name) and \
                                    ret2.comparators[0].id == param2 and isinstance(ret2.left, ast.Constant):
                                literal2 = ret2.left.value
                        if literal2 is not None:
                            # For default case: can designate vals=None or special
                            default_clause = literal2
                # If orelse contains non-Return or incorrect form, ignore
    # End of process_if

    # Start parsing:
    # Expect that func_def.body[0] is ast.If
    if func_def.body:
        first = func_def.body[0]
        if isinstance(first, ast.If):
            process_if(first)
        else:
            # possibly a single return? ignore for now
            pass

    return {
        "name": func_def.name,
        "param1": param1,
        "param2": param2,
        "clauses": clauses,  # list of ( [v1, v2, ...], literal )
        "default": default_clause  # or None
    }


def generate_z3_predicate_code(parsed):
    """
    Based on the result of parse_predicate_function, forms the text of the predicate_name_expr function:
    parsed = {
      "name": "predicate_0_2",
      "param1": "position",
      "param2": "token",
      "clauses": [
         ([0,3,4], "3"),
         ([1,2,7], "0"),
         ([5,6], "4"),
      ],
      "default": None or literal
    }
    Returns a string with the function definition:
    def predicate_0_2_expr(position, token):
        return Or(
            And(Or(position == IntVal(0), position == IntVal(3), ...), token == StringVal("3")),
            ...
            # (optional default: if default_literal is not None, we can add just token == StringVal(default) without a condition on position)
        )
    """
    name = parsed["name"] + "_expr"
    p1 = parsed["param1"]
    p2 = parsed["param2"]
    clauses = parsed["clauses"]
    default = parsed["default"]

    # Collect text
    lines = []
    header = f"def {name}({p1}, {p2}):"
    lines.append(header)
    indent = "    "
    if not clauses and default is None:
        # no constraints: always False
        lines.append(indent + "return False")
        return "\n".join(lines)

    # Start building Or(
    lines.append(indent + "return Or(")
    # For each case
    for vals, literal in clauses:
        # Build part And( Or(p1 == val1, p1 == val2, ...), p2 == literal )
        # Types: if literal is str -> StringVal, if int -> IntVal
        # Similarly for vals elements: if p1 is an Int sort variable, use IntVal;
        # but we can't check the sort of p1 here; we assume that p1 is used as Int in Z3 modeling,
        # so if vals elements are int, use IntVal, if str - StringVal.
        # Usually the first parameter position -> int, second token -> str.
        # But we don't know exactly the sort of the variable in Z3; we assume that generation was coordinated.
        # For simplicity: if element in vals has type int -> IntVal, if str -> StringVal.
        # For literal similarly.
        # Generate condition on p1:
        conds = []
        for v in vals:
            if isinstance(v, int):
                conds.append(f"{p1} == IntVal({v})")
            else:
                conds.append(f"{p1} == StringVal({v!r})")
        if len(conds) == 1:
            or_p1 = conds[0]
        else:
            or_p1 = "Or(" + ", ".join(conds) + ")"
        # condition for p2 == literal
        if isinstance(literal, int):
            cmp_p2 = f"{p2} == IntVal({literal})"
        else:
            cmp_p2 = f"{p2} == StringVal({literal!r})"
        # Result And(...)
        lines.append(indent * 2 + f"And({or_p1}, {cmp_p2}),")
    # If default is specified:
    if default is not None:
        # default: without condition on p1, only p2 == default? or vice versa.
        # The exact meaning of default needs to be agreed upon: if the original code had else: return p2==literal_default,
        # then we add just the condition p2 == literal_default here
        if isinstance(default, int):
            cmp_p2 = f"{p2} == IntVal({default})"
        else:
            cmp_p2 = f"{p2} == StringVal({default!r})"
        lines.append(indent * 2 + f"{cmp_p2},")
    # Close Or
    lines.append(indent + ")")
    return "\n".join(lines)


def generate_predicates_expr_code(src):
    funcs = extract_predicate_functions(src)
    generated = []
    for fn in funcs:
        parsed = parse_predicate_function(fn)
        if parsed is None:
            print(f"Warning: failed to parse {fn.name}, skipping")
            continue
        code = generate_z3_predicate_code(parsed)
        generated.append(code)
    return "\n\n".join(generated)


def extract_mlp_functions(source_code):
    """
    Parses source_code and returns a list of ast.FunctionDef for functions
    whose names start with "mlp_".
    """
    module = ast.parse(source_code)
    mlps = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("mlp_"):
            mlps.append(node)
    return mlps


def parse_mlp_function(func_def: ast.FunctionDef):
    """
    Takes ast.FunctionDef for mlp_*, returns:
      - conds: list of ((p_value, t_value), return_int)
      - default: int
    We assume that each return returns an integer, and keys are either int or str.
    """
    conds = []
    default = None

    def process_if(node: ast.If):
        nonlocal default
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In):
            comparator = test.comparators[0]
            if isinstance(comparator, ast.Set):
                # Collect pairs from Set.elts
                pairs = []
                for elt in comparator.elts:
                    if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                        a, b = elt.elts
                        if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                            p_val = a.value
                            t_val = b.value
                            # Support only int or str
                            if not (isinstance(p_val, (int, str)) and isinstance(t_val, (int, str))):
                                raise ValueError(f"mlp: key contains unsupported type: {type(p_val)}, {type(t_val)} in {ast.dump(elt)}")
                            pairs.append((p_val, t_val))
                        else:
                            raise ValueError(f"Expected Tuple of two Constants in set, got: {ast.dump(elt)}")
                    else:
                        raise ValueError(f"Expected Tuple in Set.elts, got: {ast.dump(elt)}")
                # Look for Return in the body of this If
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        ret = stmt.value
                        if isinstance(ret, ast.Constant) and isinstance(ret.value, int):
                            ret_val = ret.value
                        elif isinstance(ret, ast.Num):  # just in case
                            ret_val = ret.n
                        else:
                            raise ValueError(f"mlp: expected Return of an integer, got: {ast.dump(ret)}")
                        for (p_val, t_val) in pairs:
                            conds.append(((p_val, t_val), ret_val))
                        break
                # otherwise, if there's no Return in the body - can throw or skip
            else:
                raise ValueError(f"mlp: expected ast.Set in comparison key in {{...}}, got: {ast.dump(comparator)}")
        # Process elif / else
        if node.orelse:
            # if orelse is a single nested If
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                process_if(node.orelse[0])
            else:
                # search for default Return in orelse
                for stmt in node.orelse:
                    if isinstance(stmt, ast.Return):
                        ret = stmt.value
                        if isinstance(ret, ast.Constant) and isinstance(ret.value, int):
                            default = ret.value
                        elif isinstance(ret, ast.Num):
                            default = ret.n
                        # after finding default, no need to look further
                        break
        # If there's no orelse or no Return in orelse, default will remain None for now
        return

    # Look for If-chain and possible default Return
    for stmt in func_def.body:
        if isinstance(stmt, ast.If):
            process_if(stmt)
        elif isinstance(stmt, ast.Return):
            # Return outside If - possibly default
            ret = stmt.value
            if isinstance(ret, ast.Constant) and isinstance(ret.value, int):
                default = ret.value
            elif isinstance(ret, ast.Num):
                default = ret.n

    if default is None:
        raise ValueError(f"No default return found in mlp {func_def.name}")
    return conds, default


def generate_mlp_expr_code(func_def: ast.FunctionDef) -> str:
    """
    From ast.FunctionDef mlp_*, generates a string with Z3 expression code mlp_*_expr.
    Supports p_val/t_val as int or str.
    """
    name = func_def.name  # for example "mlp_1_0"
    # Extract parameters of the mlp function:
    params = [arg.arg for arg in func_def.args.args]
    if len(params) != 2:
        raise ValueError(f"mlp function {name} expected with two arguments, found: {params}")
    param0, param1 = params

    conds, default = parse_mlp_function(func_def)

    expr_name = name + "_expr"

    lines = []
    # Function header with real parameter names:
    lines.append(f"def {expr_name}({param0}, {param1}):")

    # Collect conds: we'll store as a list of tuples ((literal0, literal1), return_val),
    # where literalN is represented in code as IntVal(...) or StringVal("...")
    lines.append("    conds = [")
    for (p_val, t_val), ret_val in conds:
        # Form literals depending on type:
        # If int → IntVal, if str → StringVal(...)
        if isinstance(p_val, int):
            lit0 = f"IntVal({p_val})"
        elif isinstance(p_val, str):
            # escape quotes inside string
            esc0 = p_val.replace("\\", "\\\\").replace('"', '\\"')
            lit0 = f"StringVal(\"{esc0}\")"
        else:
            raise ValueError(f"Unsupported type p_val={p_val}({type(p_val)}) in mlp {name}")

        if isinstance(t_val, int):
            lit1 = f"IntVal({t_val})"
        elif isinstance(t_val, str):
            esc1 = t_val.replace("\\", "\\\\").replace('"', '\\"')
            lit1 = f"StringVal(\"{esc1}\")"
        else:
            raise ValueError(f"Unsupported type t_val={t_val}({type(t_val)}) in mlp {name}")

        # Add a line like: ((<lit0>, <lit1>), <ret_val>),
        lines.append(f"        (({lit0}, {lit1}), {ret_val}),")
    lines.append("    ]")
    lines.append("")
    # default
    lines.append(f"    expr = IntVal({default})  # default value")

    # Nested If cycle:
    #   for (p, t), val in reversed(conds):
    #       expr = If(And(param0 == p, param1 == t), val, expr)
    lines.append("    for (p, t), val in reversed(conds):")
    # In the comparison we need to use exactly the same param0/param1:
    #   And(param0 == p, param1 == t)
    lines.append(f"        expr = If(And({param0} == p, {param1} == t), val, expr)")
    lines.append("    return expr")

    return "\n".join(lines)


def generate_build_pipeline_from_run(run_func: ast.FunctionDef) -> str:
    if run_func is None:
        raise ValueError("Function run not found in the source code.")

    # Collect inside run:
    predicate_names = set()
    mlp_names = set()
    for node in run_func.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("predicate_"):
                predicate_names.add(node.name)
            elif node.name.startswith("mlp_"):
                mlp_names.add(node.name)

    # We'll store a single list of blocks in order of appearance
    pipeline_sequence = []
    pattern_to_layer = {}

    stmts = run_func.body
    for idx, stmt in enumerate(stmts):
        # ---- Attention block: select_closest + next aggregate ----
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "select_closest":
                # target var
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                pattern_var = stmt.targets[0].id
                # check args
                if len(call.args) != 3:
                    raise ValueError(f"select_closest with unexpected number of arguments: {ast.dump(stmt)}")
                def extract_arg_name(arg):
                    if isinstance(arg, ast.Name):
                        return arg.id
                    raise ValueError(f"Expected ast.Name in select_closest argument, got: {ast.dump(arg)}")
                keys_name = extract_arg_name(call.args[0])
                queries_name = extract_arg_name(call.args[1])
                if isinstance(call.args[2], ast.Name):
                    predicate_name = call.args[2].id
                else:
                    raise ValueError(f"Third argument of select_closest is not Name: {ast.dump(call.args[2])}")

                # check next stmt for aggregate
                if idx + 1 < len(stmts) and isinstance(stmts[idx+1], ast.Assign):
                    next_stmt = stmts[idx+1]
                    if (isinstance(next_stmt.value, ast.Call)
                        and isinstance(next_stmt.value.func, ast.Name)
                        and next_stmt.value.func.id == "aggregate"):
                        # target var
                        if len(next_stmt.targets) == 1 and isinstance(next_stmt.targets[0], ast.Name):
                            outputs_var = next_stmt.targets[0].id
                            # args: pattern_var, values
                            args = next_stmt.value.args
                            if len(args) != 2:
                                raise ValueError(f"aggregate with unexpected args: {ast.dump(next_stmt)}")
                            if not (isinstance(args[0], ast.Name) and args[0].id == pattern_var):
                                raise ValueError(f"aggregate first arg does not match pattern_var: {ast.dump(next_stmt)}")
                            if isinstance(args[1], ast.Name):
                                values_name = args[1].id
                            else:
                                raise ValueError(f"aggregate second arg is not Name: {ast.dump(args[1])}")
                            # extract layer
                            if outputs_var.startswith("attn_") and outputs_var.endswith("_outputs"):
                                layer = outputs_var[len("attn_"):-len("_outputs")]
                            else:
                                if pattern_var.startswith("attn_") and pattern_var.endswith("_pattern"):
                                    layer = pattern_var[len("attn_"):-len("_pattern")]
                                else:
                                    raise ValueError(f"Cannot extract layer name from {pattern_var} or {outputs_var}")
                            # add entry to sequence
                            pipeline_sequence.append({
                                "kind": "attn",
                                "layer": layer,
                                "keys": keys_name,
                                "queries": queries_name,
                                "predicate": predicate_name,
                                "values": values_name
                            })
                            pattern_to_layer[pattern_var] = layer
                        # otherwise ignore
                # end of attention processing
        # ---- MLP block: list comprehension ----
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.ListComp):
            lc = stmt.value
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                outputs_var = stmt.targets[0].id
            else:
                continue
            if isinstance(lc.elt, ast.Call) and isinstance(lc.elt.func, ast.Name):
                mlp_fn = lc.elt.func.id
                # expect for comprehension: one generator for zip(...)
                if len(lc.generators) == 1:
                    gen = lc.generators[0]
                    if (isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name)
                        and gen.iter.func.id == "zip"):
                        zip_args = gen.iter.args
                        if len(zip_args) == 2 and all(isinstance(a, ast.Name) for a in zip_args):
                            pos_arg = zip_args[0].id
                            input_arg = zip_args[1].id
                            if outputs_var.startswith("mlp_") and outputs_var.endswith("_outputs"):
                                layer = outputs_var[len("mlp_"):-len("_outputs")]
                            else:
                                raise ValueError(f"Cannot extract layer name from {outputs_var}")
                            pipeline_sequence.append({
                                "kind": "mlp",
                                "layer": layer,
                                "mlp_fn": mlp_fn,
                                "pos_arg": pos_arg,
                                "input_arg": input_arg
                            })
                        else:
                            raise ValueError(f"zip with unexpected args: {ast.dump(lc)}")
                    # otherwise ignore
    # After the loop: pipeline_sequence contains blocks in order of appearance.

    # === Now generating build_pipeline code ===
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
        if var_name == "tokens":
            return "tokens"
        if var_name == "positions":
            return "position_vars"
        if var_name.endswith("_outputs"):
            layer = var_name[:-len("_outputs")]
            return f'outs["{layer}"]'
        # can be extended: if other cases are needed (e.g., embeddings, etc.), insert as is
        return var_name

    # Generate blocks in the same order as in run
    for blk in pipeline_sequence:
        if blk["kind"] == "attn":
            layer = blk["layer"]
            pred_name = blk["predicate"] + "_expr"
            keys_mapped    = map_var(blk["keys"])
            queries_mapped = map_var(blk["queries"])
            values_mapped  = map_var(blk["values"])
            lines.append(
                f'    outs["attn_{layer}"] = build_attention_block(solver, '
                f'{keys_mapped}, {queries_mapped}, {blk["predicate"]}_expr, '
                f'{values_mapped}, "{layer}")'
            )
        elif blk["kind"] == "mlp":
            layer = blk["layer"]
            mlp_expr_name = blk["mlp_fn"] + "_expr"
            pos_mapped    = map_var(blk["pos_arg"])
            input_mapped  = map_var(blk["input_arg"])
            lines.append(
                f'    outs["mlp_{layer}"] = build_mlp_block(solver, '
                f'{pos_mapped}, {input_mapped}, {mlp_expr_name}, "{layer}")'
            )
        else:
            # there should be no other kinds
            pass

    # === Logits ===
    lines.append("    # === Logits ===")
    lines.append("    logits = {(i, cls): Real(f\"logit_{i}_{cls}\") for i in range(N) for cls in classes}")
    lines.append("    features = {")
    # Compose features: always have "tokens": tokens, "positions": position_vars, "ones": [IntVal(1)]*N,
    # then for each attention and mlp: "<name>_outputs": outs["<name>"]
    lines.append('        "tokens": tokens,')
    lines.append('        "positions": position_vars,')
    lines.append('        "ones": [IntVal(1)] * N,')
    # Attention/mlp outputs
    for blk in pipeline_sequence:
        if blk["kind"] == "attn":
            layer = blk["layer"]
            lines.append(f'        "attn_{layer}_outputs": outs["attn_{layer}"],')
        elif blk["kind"] == "mlp":
            layer = blk["layer"]
            lines.append(f'        "mlp_{layer}_outputs": outs["mlp_{layer}"],')
    lines.append("    }")
    # Equations for logits:
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

    # Combine into a single string:
    func_code = "\n".join(lines)
    # Re-indent for readability, although indentation is already accounted for in the lines:
    return func_code

def generate_compute_original_predictions_code() -> str:
    return '''def compute_original_predictions(input_tokens):
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
'''


def extract_run_function(tree: ast.AST):
    """
    Finds ast.FunctionDef with name 'run' in the tree. Returns AST FunctionDef.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    return None

def generate_full_z3_script(input_py_path: str, input_csv_path : str, output_py_path: str):
    """
    Main function: reads Python script input_py_path with function run(...),
    extracts necessary predicate/MLP/attention/embedding blocks and generates
    Z3 script in output_py_path.
    """
    # Read the source code
    with open(input_py_path, 'r', encoding='utf-8') as f:
        source = f.read()
    # Parse AST
    try:
        tree = ast.parse(source, filename=input_py_path)
    except SyntaxError as e:
        raise RuntimeError(f"Parsing error {input_py_path}: {e}")

    # Find the run function
    run_func = extract_run_function(tree)
    if run_func is None:
        raise RuntimeError("Function run(...) not found in the source script.")

    # Extract predicate_* and mlp_* AST functions
    attn_blocks_code = generate_predicates_expr_code(run_func)


    mlp_blocks_lines = []
    mlp_defs = extract_mlp_functions(run_func)
    for func_def in mlp_defs:
        code = generate_mlp_expr_code(func_def)
        mlp_blocks_lines.append(code)
        mlp_blocks_lines.extend(["", ""])
    mlp_blocks_code = "\n".join(mlp_blocks_lines)


    # Read static helper code
    static_code = generate_static_z3_code()

    # Generate classifier_weights reading
    weight_reading_lines = [
        "# —————— Read weights and set up constants ——————",
        'classifier_weights = pd.read_csv(\"' + input_csv_path + '\", index_col=[0, 1], dtype={\"feature\": str})',
        "classes = classifier_weights.columns.tolist()",
        ""
    ]

    # Generate build_pipeline
    build_pipeline_code = generate_build_pipeline_from_run(run_func)

    compute_predictions_code = generate_compute_original_predictions_code()
    # Combine all parts into a list of strings
    all_lines = []
    all_lines.append(static_code)
    all_lines.append("")
    all_lines.append(attn_blocks_code)
    all_lines.append(mlp_blocks_code)
    all_lines.extend(weight_reading_lines)
    all_lines.append(build_pipeline_code)
    all_lines.extend(["", ""])
    all_lines.append(compute_predictions_code)

    # Write to file
    os.makedirs(os.path.dirname(output_py_path), exist_ok=True)
    with open(output_py_path, 'w', encoding='utf-8') as fout:
        fout.write("\n".join(all_lines))

    print(f"Z3 script generated in {output_py_path}")


# ------------------
if __name__ == "main":
    generate_full_z3_script("most_freq/most_freq.py", 'most_freq/most_freq_weights.csv', "./most_freq_z3_test.py")
