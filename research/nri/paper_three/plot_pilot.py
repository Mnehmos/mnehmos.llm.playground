"""Draw every paired qualification trajectory; no pooled trial confidence bands."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    specs={
        "A":("#4d5260","Trainable, free"),"B":("#b24140","Frozen, free"),
        "C":("#087b70","Bounded"),"D":("#27569b","Causal training"),
        "E":("#576435","Isolated"),"F":("#bb7119","Extra ordinary"),
    }
    fig,axis=plt.subplots(1,2,figsize=(7.6,3.35),layout="constrained")
    for name,field,ax in [("Causal authority","authority",axis[0]),("Residual logit spread","max_residual_spread",axis[1])]:
        for arm,(color,label) in specs.items():
            for i,group in enumerate(sorted(args.root.glob(f"*-{arm}"))):
                series={}
                for file in sorted(group.glob("*.json")):
                    x=json.loads(file.read_text())
                    series[int(file.stem)]=x["metrics"]["dev"][field]
                    series.update({int(k):v["dev"][field] for k,v in x["early"].items()})
                ts=sorted(series)
                ax.plot(ts,[series[t] for t in ts],label=label if not i else None,color=color,
                        lw=1.3,alpha=.85,linestyle="--" if arm in "AE" else "-",
                        marker="." if i == 0 else None,markersize=3)
        ax.set(title=name,xlabel="AdamW updates")
        ax.spines[["top","right"]].set_visible(False)
    axis[0].set(ylim=(-.03,1.04),ylabel="Fraction of eligible swaps")
    axis[1].set_yscale("symlog",linthresh=1)
    axis[1].set(ylabel="Max within-example spread")
    axis[1].axhline(1,color=".5",ls=":",lw=1)
    axis[1].legend(frameon=False,fontsize=7,ncol=2,loc="upper left")
    fig.savefig(args.output,dpi=220)


if __name__=="__main__":
    main()
