#!/usr/bin/env python3
# LEAN corrected run: Full (KNN-SVR->refine init KNN-SVR), KNN_MICE, MICE_within. 1 SVR/condition.
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
import run_benchmark_loso as B
import run_mice_functional as MF
COMPLETE=B._find_data(); RATES=[0.10,0.20,0.30]; NITER=10
def refine(init, miss, n_iter=NITER):
    X=init.copy(); d=X.shape[1]; cm=[j for j in range(d) if miss[:,j].any()]
    for it in range(n_iter):
        for j in cm:
            mj=miss[:,j]; obs=~mj
            if obs.sum()<5: continue
            others=[c for c in range(d) if c!=j]
            br=BayesianRidge().fit(X[obs][:,others],X[obs,j]); X[mj,j]=br.predict(X[mj][:,others])
    return X
def metr(t,p,rng):
    rmse=float(np.sqrt(((p-t)**2).mean())); mae=float(np.abs(p-t).mean())
    ss=float(((t-t.mean())**2).sum()); r2=1-float(((p-t)**2).sum())/ss if ss>0 else np.nan
    return rmse,mae,(rmse/rng if rng>0 else np.nan),r2
def main(fs,fe,out):
    df=pd.read_csv(COMPLETE)
    num=[c for c in df.select_dtypes(include=["float64","int64"]).columns if c not in ("Activity","time","participant")]
    df=df.dropna(subset=num).reset_index(drop=True); parts=sorted(df.participant.unique())
    t0=time.time()
    for fi,pid in enumerate(parts):
        if fi<fs or fi>=fe: continue
        tr=df[df.participant!=pid]; ev=df[df.participant==pid].reset_index(drop=True)
        sc=StandardScaler().fit(tr[num].to_numpy(float)); Xtr=sc.transform(tr[num].to_numpy(float))
        rows=[]
        for label,gen,mech in B.SCENARIO_METHODS:
            for rate in RATES:
                mk=B.make_eval_with_missing(ev,gen,rate,1000+fi)
                Xt=ev[num].to_numpy(float); Xr=mk[num].to_numpy(float); mask=np.isnan(Xr)
                if mask.sum()==0: continue
                Xst=sc.transform(Xt); Xs=Xst.copy(); Xs[mask]=np.nan; ta=Xst[mask]
                rng=float(np.nanmax(Xst)-np.nanmin(Xst))
                cmean=np.nanmean(Xs,axis=0); im=np.where(np.isnan(Xs),cmean,Xs)
                cur,miss=MF.knn_svr_core(Xtr.copy(),Xs.copy())
                knn_out=B.imp_knn(Xtr.copy(),Xs.copy())
                outs={'Full':refine(cur.copy(),miss),'KNN_MICE':refine(knn_out.copy(),miss),'MICE_within':refine(im.copy(),miss)}
                for m,o in outs.items():
                    r=metr(ta,o[mask],rng)
                    rows.append(dict(method=m,scenario=label,mechanism=mech,rate=int(rate*100),participant=int(pid),RMSE=r[0],MAE=r[1],NRMSE=r[2],R2=r[3]))
        pd.DataFrame(rows).to_csv(out,mode="a",header=not os.path.exists(out),index=False)
        print(f"fold {fi+1} p{pid} [{time.time()-t0:.0f}s]",flush=True)
if __name__=="__main__":
    fs,fe,out=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]; main(fs,fe,out)
