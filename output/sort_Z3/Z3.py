# ========= Helper functions =========
def aggregate_expr(attn_row, values):
    expr = values[0]
    for j in reversed(range(len(attn_row))):
        expr = If(attn_row[j], values[j], expr)
    return expr


def build_attention_block(solver, keys, queries, predicate_expr, values, name):
    N = len(keys)
    attn = [[Bool(f"attn_{name}_{i}_{j}") for j in range(N)] for i in range(N)]
    any_match = [Bool(f"any_{name}_{i}") for i in range(N)]

    # check if there's any key match for each query i
    for i in range(N):
        solver.add(
            any_match[i] == Or([predicate_expr(queries[i], keys[j]) for j in range(N)])
        )

    # choose output sort based on values
    if values and isinstance(values[0], AstRef) and values[0].sort() == IntSort():
        outputs = [Int(f"attn_{name}_output_{i}") for i in range(N)]
    else:
        outputs = [String(f"attn_{name}_output_{i}") for i in range(N)]

    for i in range(N):
        # exactly one True
        solver.add(Sum([If(attn[i][j], 1, 0) for j in range(N)]) == 1)

        for j in range(N):
            if j == 0:
                # fallback only when there's no match
                solver.add(
                    Implies(attn[i][0],
                            Or(Not(any_match[i]),
                               predicate_expr(queries[i], keys[0])))
                )
            else:
                solver.add(
                    Implies(attn[i][j], predicate_expr(queries[i], keys[j]))
                )

        # closest condition
        for j in range(N):
            for k in range(N):
                solver.add(
                    Implies(
                        And(attn[i][k], predicate_expr(queries[i], keys[j])),
                        Abs(i - k) <= Abs(i - j),
                    )
                )

        # aggregate
        solver.add(outputs[i] == aggregate_expr(attn[i], values))

    return outputs


def build_mlp_block(solver, positions, tokens, mlp_expr_fn, name):
    N = len(tokens)
    outputs = [Int(f"mlp_{name}_output_{i}") for i in range(N)]
    for i in range(N):
        solver.add(outputs[i] == mlp_expr_fn(positions[i], tokens[i]))
    return outputs


def predicate_0_0_expr(pos, tok):
    return Or(
        And(pos == 0, tok == StringVal("3")),
        And(Or(pos == 1, pos == 2), tok == StringVal("1")),
        And(Or(pos == 3, pos == 4, pos == 5, pos == 6), tok == StringVal("4")),
        And(pos == 7, tok == StringVal("0"))
    )


def predicate_0_1_expr(q, k):
    return Or(
        And(Or(q == IntVal(0), q == IntVal(5), q == IntVal(7)), k == IntVal(6)),
        And(q == IntVal(1), k == IntVal(2)),
        And(Or(q == IntVal(2), q == IntVal(3)), k == IntVal(4)),
        And(q == IntVal(4), k == IntVal(5)),
        And(q == IntVal(6), k == IntVal(1)),
    )

def predicate_0_2_expr(p, t):
    return Or(
        And(Or(p == 0, p == 3, p == 4), t == StringVal("3")),
        And(Or(p == 1, p == 2, p == 7), t == StringVal("0")),
        And(Or(p == 5, p == 6), t == StringVal("4")),
    )

def predicate_0_3_expr(q, k):
    return Or(
        And(Or(q == 0, q == 1, q == 3, q == 6), k == IntVal(2)),
        And(Or(q == 2, q == 5, q == 7), k == IntVal(1)),
        And(q == 4, k == IntVal(3)),
    )


def mlp_0_0_expr(pos, token):
    key = (pos, token)
    conds = [
        # return 3
        ((0, "</s>"), 3), ((3, "</s>"), 3), ((4, "</s>"), 3), ((5, "</s>"), 3),
        ((6, "0"), 3), ((6, "1"), 3), ((6, "2"), 3), ((6, "3"), 3), ((6, "4"), 3),
        ((6, "</s>"), 3), ((6, "<s>"), 3), ((7, "2"), 3), ((7, "3"), 3),
        ((7, "4"), 3), ((7, "</s>"), 3), ((7, "<s>"), 3),
        # return 2
        ((1, "</s>"), 2), ((2, "1"), 2), ((2, "2"), 2), ((2, "3"), 2), ((2, "4"), 2),
        ((3, "1"), 2), ((3, "2"), 2), ((3, "3"), 2), ((3, "4"), 2),
        ((4, "0"), 2), ((5, "0"), 2),
        # return 0
        ((1, "0"), 0), ((1, "1"), 0), ((1, "2"), 0), ((1, "3"), 0), ((1, "<s>"), 0),
        ((2, "0"), 0), ((3, "0"), 0), ((7, "0"), 0),
        # return 1
        ((0, "0"), 1), ((1, "4"), 1)
    ]

    expr = IntVal(6)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(pos == IntVal(p), token == StringVal(t)), val, expr)
    return expr


def predicate_1_0_expr(q, k):
    return Or(
        And(q == 0, k == IntVal(1)),
        And(q == 1, k == IntVal(2)),
        And(Or(q == 2, q == 6), k == IntVal(4)),
        And(q == 3, k == IntVal(5)),
        And(q == 4, k == IntVal(6)),
        And(q == 5, k == IntVal(3)),
        And(q == 7, k == IntVal(0)),
    )

def predicate_1_1_expr(q, k):
    return Or(
        And(Or(q == 0, q == 4), k == IntVal(1)),
        And(Or(q == 1, q == 5, q == 7), k == IntVal(3)),
        And(Or(q == 2, q == 3), k == IntVal(5)),
        And(q == 6, k == IntVal(4)),
    )

def predicate_1_2_expr(p, a):
    return Or(
        And(Or(p == 0, p == 1), a == StringVal("0")),
        And(Or(p == 2, p == 3, p == 4), a == StringVal("</s>")),
        And(Or(p == 5, p == 6), a == StringVal("3")),
        And(p == 7, a == StringVal("4")),
    )

def predicate_1_3_expr(p, t):
    return Or(
        And(Or(p == 0, p == 1), t == StringVal("0")),
        And(Or(p == 2, p == 3, p == 4, p == 7), t == StringVal("</s>")),
        And(p == 5, t == StringVal("<s>")),
        And(p == 6, t == StringVal("4")),
    )


def mlp_1_0_expr(pos, token):
    conds = [
        # return 0
        ((0, "2"), 0), ((0, "3"), 0),
        ((5, "0"), 0), ((5, "1"), 0), ((5, "2"), 0), ((5, "3"), 0), ((5, "<s>"), 0),
        ((7, "3"), 0),

        # return 2
        ((1, "0"), 2), ((1, "1"), 2), ((1, "2"), 2), ((1, "3"), 2),
        ((1, "4"), 2), ((1, "</s>"), 2), ((1, "<s>"), 2),

        # return 7
        ((2, "2"), 7), ((2, "3"), 7), ((2, "4"), 7), ((2, "</s>"), 7),
        ((3, "2"), 7), ((3, "3"), 7), ((4, "2"), 7), ((4, "3"), 7),

        # return 4
        ((6, "0"), 4), ((6, "1"), 4), ((6, "<s>"), 4),
        ((7, "2"), 4), ((7, "<s>"), 4),

        # return 1
        ((6, "2"), 1), ((6, "3"), 1), ((6, "4"), 1), ((6, "</s>"), 1),
    ]

    expr = IntVal(3)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(pos == p, token == StringVal(t)), val, expr)
    return expr


import pandas as pd
from z3 import *

# —————— Read weights and set up constants ——————
classifier_weights = pd.read_csv("sort_weights.csv", index_col=[0, 1], dtype={"feature": str})
classes = classifier_weights.columns.tolist()


# —————— Pipeline: attention + MLP + logits + pred ——————
def build_pipeline(solver, tokens, position_vars):
    N = len(tokens)
    """
    Pulls all attention-, MLP-blocks, generates logits and pred[i] variables.
    Returns dictionaries outputs_by_name, logits, pred_vars.
    """
    # === Attention + MLP ===
    outs = {}
    outs["attn_0_0"] = build_attention_block(solver, tokens, position_vars, predicate_0_0_expr, tokens, "0_0")
    outs["attn_0_1"] = build_attention_block(solver, position_vars, position_vars, predicate_0_1_expr, tokens, "0_1")
    outs["attn_0_2"] = build_attention_block(solver, tokens, position_vars, predicate_0_2_expr, tokens, "0_2")
    outs["attn_0_3"] = build_attention_block(solver, position_vars, position_vars, predicate_0_3_expr, tokens, "0_3")
    outs["mlp_0_0"] = build_mlp_block(solver, position_vars, outs["attn_0_1"], mlp_0_0_expr, "0_0")

    outs["attn_1_0"] = build_attention_block(solver, position_vars, position_vars, predicate_1_0_expr, outs["attn_0_3"], "1_0")
    outs["attn_1_1"] = build_attention_block(solver, position_vars, position_vars, predicate_1_1_expr, tokens, "1_1")
    outs["attn_1_2"] = build_attention_block(solver, outs["attn_0_1"], position_vars, predicate_1_2_expr, outs["attn_0_1"], "1_2")
    outs["attn_1_3"] = build_attention_block(solver, tokens, position_vars, predicate_1_3_expr, outs["attn_0_0"], "1_3")
    outs["mlp_1_0"] = build_mlp_block(solver, position_vars, outs["attn_1_2"], mlp_1_0_expr, "1_0")

    # === Logits ===
    logits = {(i, cls): Real(f"logit_{i}_{cls}") for i in range(N) for cls in classes}
    features = {
        "tokens":           tokens,
        "positions":        position_vars,
        "ones":             [IntVal(1)] * N,
        "attn_0_0_outputs": outs["attn_0_0"],
        "attn_0_1_outputs": outs["attn_0_1"],
        "attn_0_2_outputs": outs["attn_0_2"],
        "attn_0_3_outputs": outs["attn_0_3"],
        "mlp_0_0_outputs":  outs["mlp_0_0"],
        "attn_1_0_outputs": outs["attn_1_0"],
        "attn_1_1_outputs": outs["attn_1_1"],
        "attn_1_2_outputs": outs["attn_1_2"],
        "attn_1_3_outputs": outs["attn_1_3"],
        "mlp_1_0_outputs":  outs["mlp_1_0"],
    }

    # for each (i,cls) one equation logit = Sum(If(...))
    for i in range(N):
        for cls in classes:
            contribs = []
            for feat_name, exprs in features.items():
                feat_var = exprs[i]
                for ((f_name, f_val), weights) in classifier_weights.iterrows():
                    if f_name != feat_name:
                        continue
                    w = RealVal(str(weights[cls]))
                    if feat_name == "ones":
                        contribs.append(w)
                    else:
                        if isinstance(feat_var, AstRef) and feat_var.sort() == StringSort():
                            const = StringVal(f_val)
                        else:
                            const = IntVal(int(f_val))
                        contribs.append(If(feat_var == const, w, RealVal("0")))
            solver.add(logits[(i, cls)] == Sum(contribs))

    # === Predictions ===
    pred = [String(f"pred_{i}") for i in range(N)]
    for i in range(N):
        if i == 0:
            solver.add(pred[i] == tokens[0])
        elif i == N-1:
            solver.add(pred[i] == tokens[N-1])
        else:
            for cls in classes:
                cond = And([logits[(i, cls)] >= logits[(i, o)] for o in classes if o != cls])
                solver.add(Implies(cond, pred[i] == StringVal(cls)))

    return outs, logits, pred


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

# ------- Phase 2: search for adversarial -------

def is_permutation_z3(seq1, seq2, vocab):
    """
    Returns Z3-condition that adv_tokens is a permutation of orig_tokens
    for tokens from the given vocabulary vocab (for example, ["0", "1", ..., "4"])
    """
    constraints = []
    for val in vocab:
        count_adv = Sum([If(tok == StringVal(val), 1, 0) for tok in seq1])
        count_orig = Sum([If(tok == StringVal(val), 1, 0) for tok in seq2])
        constraints.append(count_adv == count_orig)
    return And(constraints)


def find_adversarial(input_tokens, vocab):
    N = len(input_tokens)
    pred_orig = compute_original_predictions(input_tokens)
    print("Original preds:", pred_orig)

    s2 = Solver()
    # 1. New adversarial token variables
    tokens_adv = [String(f"tok_adv_{i}") for i in range(N)]
    s2.add(tokens_adv[0] == StringVal("<s>"))
    s2.add(tokens_adv[-1] == StringVal("</s>"))

    for tok in tokens_adv:
        # Only valid tokens
        s2.add(Or([tok == StringVal(v) for v in input_tokens]))

    orig_inner = input_tokens[1:-1]
    orig_z3 = [StringVal(tok) for tok in orig_inner]
    adv_inner = tokens_adv[1:-1]

    s2.add(is_permutation_z3(orig_z3, adv_inner, vocab))


    # 2. Positions (the same)
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s2.add(pos[i] == IntVal(i))


    # 3. Run pipeline on adversarial
    _, logits, pred_adv_vars = build_pipeline(s2, tokens_adv, pos)

    # 4. Prediction difference
    s2.add(Or([pred_adv_vars[i] != StringVal(pred_orig[i]) for i in range(N)]))

    if s2.check() == sat:
        m2 = s2.model()

        adv = [str(m2.evaluate(tokens_adv[i]).as_string()) for i in range(N)]
        print("Adversarial example:", adv)
        print("New preds:", [str(m2.evaluate(pred_adv_vars[i]).as_string()) for i in range(N)])
    else:
        print(f"No adversarial example found.")


# ------- Phase 3: check correctness -------

def is_sorted(seq):
    return And([
        seq[i] <= seq[i + 1] for i in range(1, len(seq) - 2)
    ])

def check_always_sorted_output(vocab, N):
    s = Solver()

    # 1. Variables: arbitrary tokens
    tokens = [String(f"tok_{i}") for i in range(N)]
    pos = [IntVal(i) for i in range(N)]

    # 2. Restrict to valid tokens from the vocabulary
    for i, tok in enumerate(tokens):
        if i == 0:
            s.add(tok == StringVal("<s>"))
        elif i == N - 1:
            s.add(tok == StringVal("</s>"))
        else:
            s.add(Or([tok == StringVal(v) for v in vocab]))

    # 3. Add pipeline and get predictions
    _, _, pred_vars = build_pipeline(s, tokens, pos)

    # 4. Prediction should be:
    #  - a permutation of the input
    #  - sorted in ascending order

    s.add(Or(
         Not(is_permutation_z3(pred_vars[1:-1], tokens[1:-1], vocab)),
         Not(is_sorted(pred_vars))
    ))

    # 5. Property verification
    if s.check() == sat:
        m = s.model()
        input_example = [str(m.evaluate(tok).as_string()) for tok in tokens]
        output_example = [str(m.evaluate(pred).as_string()) for pred in pred_vars]
        print("Counterexample found!")
        print("Input:", input_example)
        print("Output:", output_example)
        return False
    else:
        print("All possible inputs give a sorted result.")
        return True


# ------- Run -------
input_tokens = ["<s>", "1", "1", "1", "2", "1", "</s>"]
print(f"Input tokens: {input_tokens}")

vocab = ["1", "2", "3", "4"]
find_adversarial(input_tokens, vocab)

check_always_sorted_output(vocab, N=7)
