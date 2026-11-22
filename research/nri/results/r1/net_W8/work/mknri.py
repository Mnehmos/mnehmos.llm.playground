import json, hashlib
sha=hashlib.sha256(open('weights.pt','rb').read()).hexdigest()
F="[i for i in (1, 2, 3, 4) if t[i] >= 11][0]"
MAIN="(v['V_a'] + v['V_b']) % 11 if v['V_op'] == 11 else ((v['V_a'] - v['V_b']) % 11 if v['V_op'] == 12 else (v['V_a'] * v['V_b']) % 11)"
doc={
 "nri_version":"0.1",
 "model":{"id":"net_W8","weights_sha256":sha,"family":"2-layer attention-only transformer + 1 MLP on last position","condition":"ch2 constant 0","sites":[{"id":"resid_pre_mlp","dim":64}]},
 "author":{"kind":"llm","blinded":True,"model_name":"claude-opus-5-5","tool_calls":30,
   "notes":"Task: 7-token sequence [n, x1..x4, n, 14]; exactly one of positions 1-4 holds an operator token (11:+, 12:-, 13:*); output = (left neighbour) op (right neighbour) mod 11. Distractor numbers elsewhere are ignored. Operands a and b are written into a shared subspace (heads L2.2/L2.3 average the two neighbours of the operator), op-dependently, so a and b are not separately swappable; the joint (op,a,b) state lives in a ~20-dim subspace."},
 "variables":[
  {"id":"V_pos","name":"position of the operator token (1-4)","ftype":"POSITION","values":[1,2,3,4],"binding":"lambda t: "+F},
  {"id":"V_op","name":"operator token (11:+, 12:-, 13:* mod 11)","ftype":"OPERATION","values":[11,12,13],"binding":"lambda t: [t[i] for i in (1, 2, 3, 4) if t[i] >= 11][0]"},
  {"id":"V_a","name":"left operand (token immediately before operator)","ftype":"NUMBER","values":list(range(11)),"binding":"lambda t: [t[i - 1] for i in (1, 2, 3, 4) if t[i] >= 11][0]"},
  {"id":"V_b","name":"right operand (token immediately after operator)","ftype":"NUMBER","values":list(range(11)),"binding":"lambda t: [t[i + 1] for i in (1, 2, 3, 4) if t[i] >= 11][0]"},
  {"id":"V_tri","name":"joint (op, a, b) code = (op-11)*121 + a*11 + b","ftype":"OTHER","values":list(range(363)),"binding":"lambda t: [(t[i] - 11) * 121 + t[i - 1] * 11 + t[i + 1] for i in (1, 2, 3, 4) if t[i] >= 11][0]"},
  {"id":"V_res","name":"result a op b mod 11","ftype":"NUMBER","values":list(range(11)),"binding":"lambda t: [((t[i - 1] + t[i + 1]) % 11 if t[i] == 11 else ((t[i - 1] - t[i + 1]) % 11 if t[i] == 12 else (t[i - 1] * t[i + 1]) % 11)) for i in (1, 2, 3, 4) if t[i] >= 11][0]"}
 ],
 "mechanisms":[
  {"id":"M_main","component":"attn1+attn2 gather operator and its two neighbours into last position; MLP computes modular arithmetic","site_in":"resid_pre_mlp","reads":["V_op","V_a","V_b","V_pos"],"output_fn":"lambda v: "+MAIN,
   "description":"output = V_a (V_op) V_b mod 11; V_pos is read but does not affect the output"},
  {"id":"M_joint","component":"same computation, expressed over the joint operand/operator code","site_in":"resid_pre_mlp","reads":["V_tri"],
   "output_fn":"lambda v: ((v['V_tri'] // 11) % 11 + v['V_tri'] % 11) % 11 if v['V_tri'] // 121 == 0 else ((((v['V_tri'] // 11) % 11 - v['V_tri'] % 11) % 11) if v['V_tri'] // 121 == 1 else (((v['V_tri'] // 11) % 11) * (v['V_tri'] % 11)) % 11)"},
  {"id":"M_res","site_in":"resid_pre_mlp","reads":["V_res"],"output_fn":"lambda v: v['V_res']"}
 ],
 "claims":[
  {"id":"C_beh","kind":"behavior","mechanism":"M_main","predicate":[">=",0.99]},
  {"id":"C_beh_res","kind":"behavior","mechanism":"M_res","predicate":[">=",0.99]},
  {"id":"C_lr_tri","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_tri","rank":20,"mechanism":"M_joint","predicate":[">=",0.9]},
  {"id":"C_nec_tri","kind":"necessity","site":"resid_pre_mlp","variable":"V_tri","rank":20,"mechanism":"M_joint","predicate":["<=",0.2]},
  {"id":"C_lr_op","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_main","predicate":[">=",0.6]},
  {"id":"C_nec_op","kind":"necessity","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_main","predicate":["<=",0.75]},
  {"id":"C_nec_a","kind":"necessity","site":"resid_pre_mlp","variable":"V_a","rank":10,"mechanism":"M_main","predicate":["<=",0.45]},
  {"id":"C_nec_b","kind":"necessity","site":"resid_pre_mlp","variable":"V_b","rank":10,"mechanism":"M_main","predicate":["<=",0.45]},
  {"id":"C_lr_pos","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_pos","rank":3,"mechanism":"M_main","predicate":[">=",0.95]},
  {"id":"C_edit","kind":"edit","mechanism":"M_main","program":{"path":"edit_program.py","entry":"program","args":{}},
   "scope":"lambda v: v['V_op'] == 11",
   "edit_fn":"lambda v: (v['V_a'] - v['V_b']) % 11 if v['V_op'] in (11, 12) else (v['V_a'] * v['V_b']) % 11",
   "predicate":{"success":[">=",0.75],"preserve":[">=",0.98]}}
 ]
}
json.dump(doc,open('nri.json','w'),indent=1)
