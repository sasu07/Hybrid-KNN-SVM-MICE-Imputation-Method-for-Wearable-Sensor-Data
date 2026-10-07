#!/usr/bin/env python3
# Temporal baselines: LOCF (last-observation-carried-forward) and linear interpolation,
# applied ALONG THE TIME AXIS (between consecutive windows within the held-out participant).
# This is the axis the second reviewer correctly identified: rows are time-ordered windows.
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import run_benchmark_loso as B

COMPLETE=B._find_data(); RATES=[0.10,0.20,0.30]

def locf(col):
    # forward-fill then back-fill remaining leading NaNs (participant-wise, time order)
    s=pd.Series(col); return s.ffill().bfill().to_numpy()

def linterp(col):
    s=pd.Series(col); return s.interpolate(method="linear",limit_direction="both").to_numpy()

def impute_temporal(Xev_s, fn):
    out=Xev_s.copy()
    for j in range(out.shape[1]):
        out[:,j]=fn(out[:,j])
    # any all-NaN column safeguard
    out=np.where(np.isnan(out), np.nanmean(Xev_s), out)
    return out

def metrics(true,pred):
    rmse=float(np.sqrt(((pred-true)**2).mean())); mae=float(np.abs(pred-true).mean())
    ss=float(((true-true.mean())**2).sum()); r2=1-float(((pred-true)**2).sum())/ss if ss>0 else np.nan
    return rmse,mae,r2

VAR={"LOCF":lambda X: impute_temporal(X,locf),
     "LinInterp":lambda X: impute_temporal(X,linterp)}

def main(out="loso_temporal.csv"):
    df=pd.read_csv(COMPLETE)
    num=[c for c in df.select_dtypes(include=["float64","int64"]).columns if c not in ("Activity","time","participant")]
    df=df.dropna(subset=num).reset_index(drop=True); parts=sorted(df.participant.unique())
    rows=[]; t0=time.time()
    for fi,pid in enumerate(parts):
        ev=df[df.participant==pid].reset_index(drop=True)
        tr=df[df.participant!=pid]
        sc=StandardScaler().fit(tr[num].to_numpy(float))
        for label,gen,mech in B.SCENARIO_METHODS:
            for rate in RATES:
                mk=B.make_eval_with_missing(ev,gen,rate,1000+fi)
                Xt=ev[num].to_numpy(float); Xr=mk[num].to_numpy(float); mask=np.isnan(Xr)
                if mask.sum()==0: continue
                Xst=sc.transform(Xt); Xs=Xst.copy(); Xs[mask]=np.nan; ta=Xst[mask]
                for m,fn in VAR.items():
                    o=fn(Xs.copy()); rmse,mae,r2=metrics(ta,o[mask])
                    rows.append(dict(method=m,scenario=label,mechanism=mech,rate=int(rate*100),participant=int(pid),RMSE=rmse,MAE=mae,R2=r2))
        print(f"  fold {fi+1} p{pid} [{time.time()-t0:.0f}s]",flush=True)
    pd.DataFrame(rows).to_csv(out,index=False); print("saved",out,len(rows),"rows")

if __name__=="__main__": main()
