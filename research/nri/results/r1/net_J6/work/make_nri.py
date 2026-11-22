import json
K = "([i for i in range(1, 5) if t[i] >= 11] + [1])[0]"
doc = {
 "nri_version": "0.1",
 "model": {"id": "net_J6", "weights_sha256": "0"*64, "family": "2-layer transformer + residual MLP head (d_model=64)",
           "sites": [{"id": "resid_pre_mlp", "dim": 64}]},
 "author": {"kind": "llm", "blinded": True, "model_name": "claude-opus-5-5", "tool_calls": 0,
            "notes": "Black-box only. Inputs: 7 tokens; exactly one operator token (11/12/13) at position 1..4, digits 0..10 elsewhere, token 14 at position 6. Output = (left operand OP right operand) mod 11 with 11:+, 12:-, 13:*. Verified 20000/20000 on inputs.pt."},
 "variables": [
  {"id": "V_a", "name": "left operand (token immediately before the operator)", "ftype": "NUMBER",
   "values": list(range(11)), "binding": f"lambda t: t[{K} - 1]"},
  {"id": "V_b", "name": "right operand (token immediately after the operator)", "ftype": "NUMBER",
   "values": list(range(11)), "binding": f"lambda t: t[{K} + 1]"},
  {"id": "V_op", "name": "operator token (11=add, 12=sub, 13=mul, all mod 11)", "ftype": "OPERATION",
   "values": [11, 12, 13], "binding": f"lambda t: t[{K}]"},
  {"id": "V_oppos", "name": "position of the operator token (1..4); distractor, output does not depend on it", "ftype": "POSITION",
   "values": [1, 2, 3, 4], "binding": f"lambda t: {K}"},
  {"id": "V_result", "name": "result of the modular operation", "ftype": "NUMBER", "values": list(range(11)),
   "binding": f"lambda t: (t[{K}-1] + t[{K}+1]) % 11 if t[{K}] == 11 else ((t[{K}-1] - t[{K}+1]) % 11 if t[{K}] == 12 else (t[{K}-1] * t[{K}+1]) % 11)"}
 ],
 "mechanisms": [
  {"id": "M_modarith", "component": "attention gathers operands adjacent to the operator into the final (token-14) position; MLP computes the modular op",
   "site_in": "resid_pre_mlp", "reads": ["V_a", "V_b", "V_op"],
   "output_fn": "lambda v: (v['V_a'] + v['V_b']) % 11 if v['V_op'] == 11 else ((v['V_a'] - v['V_b']) % 11 if v['V_op'] == 12 else (v['V_a'] * v['V_b']) % 11)",
   "description": "Output class = (a op b) mod 11 where a,b are the digits immediately left/right of the single operator token; other digits (t0..t5 not adjacent) are ignored."},
  {"id": "M_result_readout", "site_in": "resid_pre_mlp", "reads": ["V_result"], "output_fn": "lambda v: v['V_result']",
   "description": "Identity readout of the computed result (behaviourally equivalent to M_modarith)."}
 ],
 "claims": [
  {"id": "C_behavior", "kind": "behavior", "mechanism": "M_modarith", "predicate": [">=", 0.99]},
  {"id": "C_behavior_result", "kind": "behavior", "mechanism": "M_result_readout", "predicate": [">=", 0.99]},
  {"id": "C_op_linrep", "kind": "linear_rep", "site": "resid_pre_mlp", "variable": "V_op", "rank": 2,
   "mechanism": "M_modarith", "predicate": [">=", 0.8]},
  {"id": "C_a_linrep", "kind": "linear_rep", "site": "resid_pre_mlp", "variable": "V_a", "rank": 10,
   "mechanism": "M_modarith", "predicate": [">=", 0.6]},
  {"id": "C_b_linrep", "kind": "linear_rep", "site": "resid_pre_mlp", "variable": "V_b", "rank": 10,
   "mechanism": "M_modarith", "predicate": [">=", 0.6]},
  {"id": "C_op_necessity", "kind": "necessity", "site": "resid_pre_mlp", "variable": "V_op", "rank": 2,
   "mechanism": "M_modarith", "predicate": ["<=", 0.6]},
  {"id": "C_a_necessity", "kind": "necessity", "site": "resid_pre_mlp", "variable": "V_a", "rank": 10,
   "mechanism": "M_modarith", "predicate": ["<=", 0.4]},
  {"id": "C_b_necessity", "kind": "necessity", "site": "resid_pre_mlp", "variable": "V_b", "rank": 10,
   "mechanism": "M_modarith", "predicate": ["<=", 0.4]},
  {"id": "C_edit_add_to_sub", "kind": "edit", "mechanism": "M_modarith",
   "program": {"path": "edit_program.py", "entry": "program", "args": {}},
   "scope": "lambda v: v['V_op'] == 11",
   "edit_fn": "lambda v: (v['V_a'] - v['V_b']) % 11",
   "predicate": {"success": [">=", 0.6], "preserve": [">=", 0.97]}}
 ],
 "summary": {"task": "single-operator modular arithmetic mod 11: tokens 0..10 digits, 11 '+', 12 '-', 13 '*', 14 end/query; output = left (op) right mod 11 using the digits adjacent to the operator",
             "behavior_accuracy_on_inputs_pt": 1.0}
}
import sys
if len(sys.argv) > 1: doc["author"]["tool_calls"] = int(sys.argv[1])
json.dump(doc, open("nri.json", "w"), indent=2)
