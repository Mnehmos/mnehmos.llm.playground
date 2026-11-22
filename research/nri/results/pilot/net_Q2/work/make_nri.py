import json, hashlib
sha=hashlib.sha256(open('weights.pt','rb').read()).hexdigest()
K="[i for i in range(len(t)) if t[i] >= 11 and t[i] <= 13][0]"
doc={
 "nri_version":"0.1",
 "model":{"id":"net_Q2","weights_sha256":sha,"family":"2-layer bidirectional attention + 1 MLP, readout at last position",
          "sites":[{"id":"resid_pre_mlp","dim":64}]},
 "author":{"kind":"llm","blinded":True,"model_name":"claude-opus-5-5","tool_calls":36,
   "notes":"Task: modular arithmetic mod 11. The unique token in {11,12,13} is the operator (+,-,*); a = token before it, b = token after it; output = (a op b) mod 11. ch2 marks the roles (1=a, 3=op, 2=b, 4=query position, 0=filler)."},
 "variables":[
  {"id":"V_op","name":"operator token (11:+, 12:-, 13:*)","ftype":"OPERATION","values":[11,12,13],
   "binding":"lambda t: [x for x in t if x >= 11 and x <= 13][0]"},
  {"id":"V_oppos","name":"position of the operator token","ftype":"POSITION","values":[1,2,3,4],
   "binding":"lambda t: "+K},
  {"id":"V_a","name":"left operand (token before operator)","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: t[[i for i in range(len(t)) if t[i] >= 11 and t[i] <= 13][0] - 1]"},
  {"id":"V_b","name":"right operand (token after operator)","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: t[[i for i in range(len(t)) if t[i] >= 11 and t[i] <= 13][0] + 1]"},
  {"id":"V_res","name":"(a op b) mod 11","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: (lambda k: ((t[k-1] + t[k+1]) if t[k] == 11 else ((t[k-1] - t[k+1]) if t[k] == 12 else (t[k-1] * t[k+1]))) % 11)("+K+")"}
 ],
 "mechanisms":[
  {"id":"M_modarith","component":"attn1+attn2 gather (a, op, b) into the last position; MLP computes the result",
   "site_in":"resid_pre_mlp","reads":["V_op","V_a","V_b"],
   "output_fn":"lambda v: ((v['V_a'] + v['V_b']) % 11) if v['V_op'] == 11 else (((v['V_a'] - v['V_b']) % 11) if v['V_op'] == 12 else ((v['V_a'] * v['V_b']) % 11))",
   "description":"Output class = (a OP b) mod 11 with OP = +, -, * for tokens 11, 12, 13. Tokens 0-10 are digits; token 14 is a constant query token at position 6."}
 ],
 "claims":[
  {"id":"C_behavior","kind":"behavior","mechanism":"M_modarith","predicate":[">=",0.99]},
  {"id":"C_lin_op","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_modarith","predicate":[">=",0.8]},
  {"id":"C_lin_a","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_a","rank":10,"mechanism":"M_modarith","predicate":[">=",0.65]},
  {"id":"C_lin_b","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_b","rank":10,"mechanism":"M_modarith","predicate":[">=",0.65]},
  {"id":"C_nec_op","kind":"necessity","site":"resid_pre_mlp","variable":"V_op","rank":2,"mechanism":"M_modarith","predicate":[">=",0.95],
   "notes":"Negative necessity claim: removing the best rank-2 linear operator subspace leaves task accuracy essentially intact (operator identity is also carried nonlinearly in the operand features)."},
  {"id":"C_edit_plus_to_minus","kind":"edit","mechanism":"M_modarith",
   "program":{"path":"edit_program.py","entry":"program","args":{}},
   "scope":"lambda v: v['V_op'] == 11",
   "edit_fn":"lambda v: (v['V_a'] - v['V_b']) % 11",
   "predicate":{"success":[">=",0.85],"preserve":[">=",0.96]}}
 ],
 "summary":{"task":"(a op b) mod 11, op in {+,-,*} given by token 11/12/13; operands flank the operator",
            "edit":"W_in-only closed-form edit turning + into -: ~92% success, ~98% preservation on held-out data in my tests"}
}
json.dump(doc,open('nri.json','w'),indent=2)
