import argparse, re
from pathlib import Path
import pandas as pd

LOC = {
"basal_ganglia":["basal ganglia","bg"],"corona_radiata":["corona radiata"],
"centrum_semiovale":["centrum semiovale"],"internal_capsule":["internal capsule","posterior limb"],
"thalamus":["thalamus","thalamic"],"hippocampus":["hippocampus"],"caudate":["caudate"],
"frontal_lobe":["frontal"],"parietal_lobe":["parietal"],"temporal_lobe":["temporal"],
"occipital_lobe":["occipital"],"frontoparietal":["frontoparietal","fronto parietal"],
"temporoparietal":["temporoparietal","temporo parietal"],"precentral_gyrus":["precentral"],
"postcentral_gyrus":["postcentral"],"mca_territory":["mca territory"],"aca_territory":["aca territory"],
"pca_territory":["pca territory"],"pons":["pons","pontine"],"medulla":["medulla","medullary"],
"midbrain":["midbrain"],"cerebellum":["cerebellum","cerebellar"],"vermis":["vermis"],
"corpus_callosum":["corpus callosum"],"cortex":["cortical","cortex"]
}

def norm(s):
    s=str(s).lower().replace("&"," and ")
    s=re.sub(r"[^a-z0-9]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def lat(s):
    t=norm(s)
    l=bool(re.search(r"\bleft\b",t)); r=bool(re.search(r"\bright\b",t))
    b=bool(re.search(r"\bboth\b|\bbilateral\b",t))
    if b or (l and r): return {"left","right"}
    if l: return {"left"}
    if r: return {"right"}
    return set()

def locs(s):
    t=norm(s); out=set()
    for k,ps in LOC.items():
        if any(norm(p) in t for p in ps): out.add(k)
    if "frontoparietal" in out: out-={"frontal_lobe","parietal_lobe"}
    if "temporoparietal" in out: out-={"temporal_lobe","parietal_lobe"}
    return out

def tuples(s):
    la=lat(s); lo=locs(s)
    return {(a,b) for a in la for b in lo} if la and lo else set()

def f1(p,t):
    p=set(p); t=set(t)
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    inter=len(p&t)
    if inter==0: return 0.0
    pr=inter/len(p); rc=inter/len(t)
    return 2*pr*rc/(pr+rc)

def cols(df):
    pred=next((c for c in ["prediction","pred","generated_report","output"] if c in df.columns),None)
    tgt=next((c for c in ["target","target_report","reference","acute_target_sentence"] if c in df.columns),None)
    cid=next((c for c in ["case_id","id","patient_id"] if c in df.columns),None)
    if pred is None or tgt is None: raise ValueError(df.columns.tolist())
    return cid,pred,tgt

def group(tt):
    l={a for a,b in tt}; lo={b for a,b in tt}
    if len(l)>=2: return "bilateral_or_multifocal"
    if len(lo)>=2: return "multi_location"
    return "single_location"

def audit(path,out_dir):
    df=pd.read_csv(path).fillna("")
    cid,pc,tc=cols(df)
    rows=[]
    for _,r in df.iterrows():
        target=str(r[tc]); pred=str(r[pc])
        tl,pl=lat(target),lat(pred)
        tloc,ploc=locs(target),locs(pred)
        tt,pt=tuples(target),tuples(pred)
        rows.append({
            "case_id":str(r[cid]) if cid else "",
            "patient_id":str(r["patient_id"]) if "patient_id" in df.columns else (str(r[cid]) if cid else ""),
            "target":target,"prediction":pred,
            "target_laterality":"|".join(sorted(tl)),"pred_laterality":"|".join(sorted(pl)),
            "target_locations":"|".join(sorted(tloc)),"pred_locations":"|".join(sorted(ploc)),
            "target_tuples":"|".join(f"{a}:{b}" for a,b in sorted(tt)),
            "pred_tuples":"|".join(f"{a}:{b}" for a,b in sorted(pt)),
            "laterality_exact":int(tl==pl and len(tl)>0),
            "location_f1":f1(ploc,tloc),
            "location_exact":int(ploc==tloc and len(tloc)>0),
            "tuple_f1":f1(pt,tt),
            "tuple_exact":int(pt==tt and len(tt)>0),
            "group":group(tt),
            "pred_len":len(norm(pred).split()),
            "target_len":len(norm(target).split()),
        })
    od=Path(out_dir); od.mkdir(parents=True,exist_ok=True)
    case=pd.DataFrame(rows)
    name=Path(path).stem
    case.to_csv(od/f"{name}_tuple_audit_cases.csv",index=False)
    mets=[]
    groups=[("all",case)]+list(case.groupby("group"))
    for gname,g in groups:
        mets.append({
            "file":str(path),"group":gname,"n":len(g),
            "laterality_exact":g.laterality_exact.mean(),
            "location_f1":g.location_f1.mean(),
            "location_exact":g.location_exact.mean(),
            "tuple_f1":g.tuple_f1.mean(),
            "tuple_exact":g.tuple_exact.mean(),
            "unique_predictions":g.prediction.nunique(),
            "top1_prediction_ratio":g.prediction.value_counts(normalize=True).iloc[0] if len(g) else 0,
            "pred_len":g.pred_len.mean(),"target_len":g.target_len.mean(),
        })
    sm=pd.DataFrame(mets)
    sm.to_csv(od/f"{name}_tuple_audit_summary.csv",index=False)
    print("\n===",path,"===")
    print(sm.to_string(index=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",nargs="+",required=True)
    ap.add_argument("--out_dir",default="outputs/report_metrics")
    a=ap.parse_args()
    for p in a.csv: audit(p,a.out_dir)

if __name__=="__main__": main()
