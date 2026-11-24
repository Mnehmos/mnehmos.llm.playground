from pathlib import Path
import csv,json
root=Path('conversion-run')
rows=[]
for ck in sorted(root.glob('step_*')):
 if not (ck/'review.json').is_file():continue
 e=json.loads((ck/'evidence.json').read_text());r=json.loads((ck/'review.json').read_text())
 c=e['concepts'];q=c['conversion']
 row={'step':e['step'],'action':r['action'],'prediction_result':r.get('previous_resolution',{}).get('status','initial'),
      'training_wall_s':e['training_wall_s'],'instrument_wall_s':e['instrument_wall_s']}
 for name,v in q.items():row[name]=v.get('correct',v.get('intended'))/v['n']
 for name,v in c['dev_acc'].items():row[name]=v
 rows.append(row)
with (root/'trajectory.csv').open('w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('audits',len(rows),'last',rows[-1]['step'])
