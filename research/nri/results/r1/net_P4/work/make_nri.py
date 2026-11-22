import json
P="[i for i in range(1,5) if t[i]>=11][0]"
F3="((v['V_a']+v['V_b'])%11 if v['V_op']==11 else ((v['V_a']-v['V_b'])%11 if v['V_op']==12 else (v['V_a']*v['V_b'])%11))"
doc={
 "nri_version":"0.1",
 "model":{"id":"net_P4","weights_sha256":"b6298ec49bcc6e9d74e202c18c31538cd312a333db8bb007dc05218d35eb356e",
          "family":"2-layer attention-only transformer + final-position MLP","condition":"ch2 constant 0 in data",
          "sites":[{"id":"resid_pre_mlp","dim":64}]},
 "author":{"kind":"llm","blinded":True,"model_name":"claude-opus-5-5","tool_calls":45,
   "notes":"Task: modular arithmetic mod 11 on a single infix binary expression 'a OP b' embedded at a random place in a 6-token sequence of digits 0..10, OP in {11:+,12:-,13:*}, token 14 = query/end. Behavior fit 100%. Operand code at resid_pre_mlp is entangled with the operator (a,b not individually swappable); the joint (a,b,OP) code lives in a ~16-18 dim subspace. Edit claim is deliberately weak: a W_in-only affine edit cannot implement +->- here (see report)."},
 "variables":[
  {"id":"V_oppos","name":"position of the operator token (1..4)","ftype":"POSITION","values":[1,2,3,4],"binding":"lambda t: "+P},
  {"id":"V_op","name":"operator token (11:+, 12:-, 13:*)","ftype":"OPERATION","values":[11,12,13],"binding":"lambda t: t["+P+"]"},
  {"id":"V_a","name":"left operand (token before operator)","ftype":"NUMBER","values":list(range(11)),"binding":"lambda t: t["+P+"-1]"},
  {"id":"V_b","name":"right operand (token after operator)","ftype":"NUMBER","values":list(range(11)),"binding":"lambda t: t["+P+"+1]"},
  {"id":"V_abop","name":"joint code (a,b,op) = 33*a + 3*b + (op-11)","ftype":"OTHER","binding":"lambda t: 33*t["+P+"-1] + 3*t["+P+"+1] + (t["+P+"]-11)"},
  {"id":"V_result","name":"(a op b) mod 11","ftype":"NUMBER","values":list(range(11)),
   "binding":"lambda t: (lambda a,b,o: (a+b)%11 if o==11 else ((a-b)%11 if o==12 else (a*b)%11))(t["+P+"-1], t["+P+"+1], t["+P+"])"}
 ],
 "mechanisms":[
  {"id":"M_main","component":"attn1+attn2 gather a, OP, b to last position; MLP computes the modular op","site_in":"resid_pre_mlp",
   "reads":["V_a","V_b","V_op"],"output_fn":"lambda v: "+F3,
   "description":"output = (a OP b) mod 11 where OP is the single operator token and a,b its immediate neighbours"},
  {"id":"M_joint","site_in":"resid_pre_mlp","reads":["V_abop"],
   "output_fn":"lambda v: (lambda a,b,o: (a+b)%11 if o==0 else ((a-b)%11 if o==1 else (a*b)%11))(v['V_abop']//33, (v['V_abop']//3)%11, v['V_abop']%3)",
   "description":"same function, read from the joint (a,b,op) code"},
  {"id":"M_result","site_in":"resid_pre_mlp","reads":["V_result"],"output_fn":"lambda v: v['V_result']"}
 ],
 "claims":[
  {"id":"C_behavior","kind":"behavior","mechanism":"M_main","predicate":[">=",0.99]},
  {"id":"C_rep_abop","kind":"linear_rep","site":"resid_pre_mlp","variable":"V_abop","rank":18,"mechanism":"M_joint","predicate":[">=",0.95]},
  {"id":"C_nec_abop","kind":"necessity","site":"resid_pre_mlp","variable":"V_abop","rank":18,"mechanism":"M_joint","predicate":["<=",0.2]},
  {"id":"C_nec_result","kind":"necessity","site":"resid_pre_mlp","variable":"V_result","rank":10,"mechanism":"M_result","predicate":["<=",0.25]},
  {"id":"C_nec_a","kind":"necessity","site":"resid_pre_mlp","variable":"V_a","rank":10,"mechanism":"M_main","predicate":["<=",0.45]},
  {"id":"C_nec_b","kind":"necessity","site":"resid_pre_mlp","variable":"V_b","rank":10,"mechanism":"M_main","predicate":["<=",0.45]},
  {"id":"C_edit","kind":"edit","mechanism":"M_main","program":{"path":"edit_program.py","entry":"program","args":{}},
   "scope":"lambda v: v['V_op'] == 11","edit_fn":"lambda v: (v['V_a'] - v['V_b']) % 11",
   "predicate":{"success":[">=",0.08],"preserve":[">=",0.97]}}
 ],
 "summary":{"task":"single binary modular expression (a OP b) mod 11, OP in {+,-,*}","behavior_fit":1.0,
   "edit":"weak: near-null W_in edit (success ~0.11 incl. trivial b=0 cases, preserve ~0.99); faithful +->- edit not achievable by an affine W_in change in this network"}
}
json.dump(doc,open('nri.json','w'),indent=2)
