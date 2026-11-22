import json
OPI="[i for i in range(1, 5) if t[i] in (11, 12, 13)]"
doc={
 "nri_version":"0.1",
 "model":{"id":"net_B3","weights_sha256":"0"*64,"family":"2-layer transformer + residual MLP head (d_model=64)",
          "condition":"blind black-box","sites":[{"id":"resid_pre_mlp","dim":64}]},
 "author":{"kind":"llm","blinded":True,"model_name":"claude-opus-5-5","tool_calls":36,
   "notes":"Black-box only. Input: 6 slots + end token 14; exactly one operator token (11,12,13) at a position 1..4, numbers 0..10 elsewhere. Output = (left neighbour OP right neighbour) mod 11 with 11:+ 12:- 13:*; 100% agreement on the 20000 provided inputs. Other numbers are distractors."},
 "variables":[
  {"id":"V_op","name":"operator (0:+ 1:- 2:*)","ftype":"OPERATION","values":[0,1,2],
   "binding":"lambda t: ([t[i] for i in range(1, 5) if t[i] in (11, 12, 13)] + [11])[0] - 11"},
  {"id":"V_a","name":"left operand (token immediately before the operator)","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: t[([i for i in range(1, 5) if t[i] in (11, 12, 13)] + [1])[0] - 1] % 11"},
  {"id":"V_b","name":"right operand (token immediately after the operator)","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: t[([i for i in range(1, 5) if t[i] in (11, 12, 13)] + [1])[0] + 1] % 11"},
  {"id":"V_oppos","name":"position of the operator token","ftype":"POSITION","values":[1,2,3,4],
   "binding":"lambda t: ([i for i in range(1, 5) if t[i] in (11, 12, 13)] + [1])[0]"},
  {"id":"V_result","name":"result (a op b) mod 11","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: (lambda o: [(t[o-1] + t[o+1]) % 11, (t[o-1] - t[o+1]) % 11, (t[o-1] * t[o+1]) % 11][(t[o] - 11) % 3])(([i for i in range(1, 5) if t[i] in (11, 12, 13)] + [1])[0])"}
 ],
 "mechanisms":[
  {"id":"M_main","component":"attention gathers operator and its two neighbours into the last position; MLP computes the mod-11 table",
   "site_in":"resid_pre_mlp","reads":["V_op","V_a","V_b"],
   "output_fn":"lambda v: [(v['V_a'] + v['V_b']) % 11, (v['V_a'] - v['V_b']) % 11, (v['V_a'] * v['V_b']) % 11][v['V_op']]",
   "description":"Modular arithmetic mod 11 on the two numbers adjacent to the single operator token; 11=+, 12=-, 13=*."},
  {"id":"M_result","site_in":"resid_pre_mlp","reads":["V_result"],"output_fn":"lambda v: v['V_result']",
   "description":"Same function expressed through the derived result variable."}
 ],
 "claims":[
  {"id":"C_behavior","kind":"behavior","mechanism":"M_main","predicate":[">=",0.99]},
  {"id":"C_behavior_result","kind":"behavior","mechanism":"M_result","predicate":[">=",0.99]},
  {"id":"C_op_linear","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_main","predicate":[">=",0.5]},
  {"id":"C_op_necessity","kind":"necessity","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_main","predicate":["<=",0.8]},
  {"id":"C_a_necessity","kind":"necessity","site":"resid_pre_mlp","variable":"V_a","rank":10,"mechanism":"M_main","predicate":["<=",0.5]},
  {"id":"C_b_necessity","kind":"necessity","site":"resid_pre_mlp","variable":"V_b","rank":10,"mechanism":"M_main","predicate":["<=",0.5]},
  {"id":"C_edit_plus_to_minus","kind":"edit","mechanism":"M_main",
   "program":{"path":"edit_program.py","entry":"program","args":{}},
   "scope":"lambda v: v['V_op'] == 0",
   "edit_fn":"lambda v: (v['V_a'] - v['V_b']) % 11",
   "predicate":{"success":[">=",0.5],"preserve":[">=",0.95]}}
 ],
 "summary":{"task":"y = (a OP b) mod 11 where OP in {+,-,*} is the single operator token (11,12,13) at position k in 1..4, a = t[k-1], b = t[k+1]; remaining numbers are distractors; t[6]=14 is an end token.",
            "evidence":"exhaustive purity search over position pairs per operator position/type; full 11x11 tables recovered; 20000/20000 agreement"}
}
json.dump(doc,open('nri.json','w'),indent=2)
