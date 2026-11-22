import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/tri.py').read().split("for r in [10,20,30")[0])
for r in [14,16,18]: print('tri',r,ich('tri',r))
