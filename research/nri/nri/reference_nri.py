"""Reference NRI document for a zoo model (ground-truth bindings, known to the authors).

Used to (1) validate the executor end to end and (2) serve as the answer key the
blinded-interpreter pilot is scored against. A blinded interpreter must produce its
own bindings; it is never shown this file.
"""
from __future__ import annotations

from . import tasks

OPLO, OPHI = tasks.OP_TOKENS[0], tasks.OP_TOKENS[-1]
OPI = f"[i for i, x in enumerate(t) if {OPLO} <= x <= {OPHI}][0]"
OUT_FN = ("lambda v: (v['V_a'] + v['V_b']) % 11 if v['V_op'] == 0 else "
          "((v['V_a'] - v['V_b']) % 11 if v['V_op'] == 1 else (v['V_a'] * v['V_b']) % 11)")


def reference_doc(model_id: str, weights_sha: str, condition: str, program_path: str) -> dict:
    variables = [
        {"id": "V_op", "name": "operation", "ftype": "OPERATION", "values": list(tasks.OPS),
         "binding": f"lambda t: t[{OPI}] - {OPLO}"},
        {"id": "V_a", "name": "left operand", "ftype": "NUMBER",
         "binding": f"lambda t: t[{OPI} - 1]"},
        {"id": "V_b", "name": "right operand", "ftype": "NUMBER",
         "binding": f"lambda t: t[{OPI} + 1]"},
    ]
    claims = [{"id": "C_behavior", "kind": "behavior", "mechanism": "M_arith",
               "predicate": [">=", 0.99]}]
    for v, r in (("V_op", 2), ("V_a", 10), ("V_b", 10)):
        claims.append({"id": f"C_rep_{v}", "kind": "linear_rep", "site": "resid_pre_mlp",
                       "variable": v, "rank": r, "mechanism": "M_arith",
                       "predicate": [">=", 0.9]})
        claims.append({"id": f"C_nec_{v}", "kind": "necessity", "site": "resid_pre_mlp",
                       "variable": v, "rank": r, "mechanism": "M_arith",
                       "predicate": ["<=", 0.5]})
    claims.append({
        "id": "C_edit_add_to_sub", "kind": "edit", "mechanism": "M_arith",
        "program": {"path": program_path, "entry": "program",
                    "args": {"var": "V_op", "x": 0, "y": 1}},
        "scope": "lambda v: v['V_op'] == 0",
        "edit_fn": ("lambda v: (v['V_a'] - v['V_b']) % 11 if v['V_op'] in (0, 1) "
                    "else (v['V_a'] * v['V_b']) % 11"),
        "predicate": {"success": [">=", 0.9], "preserve": [">=", 0.97]}})
    return {
        "nri_version": "0.1",
        "model": {"id": model_id, "weights_sha256": weights_sha, "family": "TinyTyped",
                  "condition": condition, "sites": [{"id": "resid_pre_mlp", "dim": 64}]},
        "author": {"kind": "reference", "blinded": False},
        "variables": variables,
        "mechanisms": [{"id": "M_arith", "component": "mlp", "reads": ["V_op", "V_a", "V_b"],
                        "site_in": "resid_pre_mlp", "output_fn": OUT_FN,
                        "description": "MLP computes OP(a, b) mod 11 from the gathered variables"}],
        "claims": claims,
    }
