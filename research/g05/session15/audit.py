from pair import *
import pandas as pd

def main():
    a=pd.read_csv(R/'scores.csv'); b=pd.read_csv(R/'pair_scores.csv'); n=0; k=0; locks=json.loads((R/'PARENT_LOCK.json').read_text()); maximum=0.; parents=0
    def cmp(d,row,keys):
        nonlocal maximum
        for x,y in keys.items():
            diff=abs(d[x]-float(row[y])); maximum=max(maximum,diff); assert diff<1e-6,(x,y,d[x],row[y])
    for s,mode,step,path in paths():
        if step!=6000:continue
        assert sha(path)==locks[str(path.relative_to(P))]; parents+=1; name=f'{s}_{mode}'
        if not (R/f'{name}_panel.pt').exists():continue
        m=TinyTyped('mass_only',**MODEL_KW); init=t.load(path,weights_only=True)['model'];m.load_state_dict(init);m.eval(); p=t.load(R/f'{name}_panel.pt',weights_only=True);ba=T.build(aslat(p['z']));x,y=ba.tokens,ba.y
        for row in a[(a.backbone==name)&(a.step==20)].to_dict('records'):
            ck=t.load(R/f'{name}_L{row["layer"]}H{row["head"]}_{row["arm"]}.pt',weights_only=True); assert ck['parent_sha']==sha(path);m.load_state_dict(ck['model']);ps=params(m,row['layer'],row['head']);allowed={id(p):k for p,k in ps}
            for key,p0 in m.named_parameters():
                diff=p0.detach()-init[key];
                if id(p0) in allowed:diff=diff.clone();diff[allowed[id(p0)]]=0
                assert not diff.count_nonzero(),key
            e=score(m,x[p['exam']],y[p['exam']]);r=score(m,x[p['retention']],y[p['retention']]);cmp(e,row,dict(sign='exam_sign',loss='exam_ce',exact='exam_exact',quotient='exam_quotient'));cmp(r,row,dict(exact='retention_exact',quotient='retention_quotient',loss='retention_ce'));n+=1
        p=t.load(R/'pair_states'/f'{name}_panel.pt',weights_only=True);ba=T.build(aslat(p['z']));x,y=ba.tokens,ba.y
        for row in b[(b.backbone==name)&(b.step>0)].to_dict('records'):
            m.load_state_dict(init);ck=t.load(R/'pair_states'/f'{name}_L{row["layer"]}H{row["head"]}_{row["arm"]}_{row["step"]}.pt',weights_only=True);assert ck['parent_sha']==sha(path);ps=params(m,row['layer'],row['head']);
            with t.no_grad():
                for (p0,ix),w in zip(ps,ck['weights']):p0[ix]=w
            e=pair_metrics(m,x[p['exam']],y[p['exam']]);cmp(e,row,{key:key for key in e});k+=1
    result=dict(parents_verified=parents,full_final_states_verified=n,sparse_horizon_states_verified=k,max_metric_absolute_difference=maximum,scope='reload and score reproduction, not full training replays');(R/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
