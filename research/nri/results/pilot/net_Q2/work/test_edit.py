import torch,sys,time; sys.path.insert(0,'.'); sys.path.insert(0,'work')
from mock import Host, evaluate
import edit_program
host=Host()
for br in [0,3]:
    t0=time.time(); dW,db=edit_program.program(host,None,boost_rounds=br)
    print(br,dW.shape,db.shape,evaluate(dW,db,host,nt=20000),round(time.time()-t0,1))
