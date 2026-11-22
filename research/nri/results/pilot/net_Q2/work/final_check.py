import torch,sys,json,time; sys.path.insert(0,'.'); sys.path.insert(0,'work')
from mock import Host, evaluate
import edit_program
host=Host(); doc=json.load(open('nri.json'))
t0=time.time(); dW,db=edit_program.program(host,doc,**doc['claims'][-1]['program']['args'])
print(tuple(dW.shape),tuple(db.shape),dW.dtype,'success/preserve (last 6000 rows)',evaluate(dW,db,host,nt=6000),round(time.time()-t0,1),'s')
