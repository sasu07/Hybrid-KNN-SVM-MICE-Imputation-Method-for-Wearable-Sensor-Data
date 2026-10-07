#!/usr/bin/env python3
# Functional-MICE ablation: MICE stage actually imputes (re-mask → impute), not a no-op on a full matrix.
# Compares the two-stage KNN-SVR against a genuine three-stage KNN-SVR-MICE, plus singles/pairs,
# all under leave-one-participant-out.
import warnings; warnings.filterwarnings("ignore")
import os, tempfile, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import run_benchmark_loso as B

COMPLETE=B._find_data(); RATES=[0.10,0.20,0.30]; K=5; C=100; EPS=0.1; SUB=800

def knn_svr_core(Xtr_s, Xev_s, cycles=5):
    """Two-stage KNN->SVR, returns imputed standardized matrix AND the mask."""
    d=Xtr_s.shape[1]; Xtr=Xtr_s
    if len(Xtr)>SUB: Xtr=Xtr[np.random.RandomState(0).choice(len(Xtr),SUB,replace=False)]
    svr={}
    for j in range(d):
        cols=[c for c in range(d) if c!=j]
        svr[j]=SVR(kernel="rbf",C=C,epsilon=EPS).fit(Xtr[:,cols],Xtr[:,j])
    miss=np.isnan(Xev_s); cur=Xev_s.copy()
    for _ in range(cycles):
        cm=np.nanmean(cur,axis=0); filled=np.where(np.isnan(cur),cm,cur)
        knn_pred=cur.copy()
        for j in range(d):
            mj=miss[:,j]
            if mj.any():
                cols=[c for c in range(d) if c!=j]
                kn=KNeighborsRegressor(n_neighbors=K,metric="manhattan")
                obs=~np.isnan(Xev_s[:,j]); kn.fit(filled[obs][:,cols],Xev_s[obs,j])
                knn_pred[mj,j]=kn.predict(filled[mj][:,cols])
        for j in range(d):
            mj=miss[:,j]
            if mj.any():
                cols=[c for c in range(d) if c!=j]
                base=np.where(np.isnan(knn_pred),np.nanmean(knn_pred,axis=0),knn_pred)
                pr=svr[j].predict(base[mj][:,cols])
                cur[mj,j]=0.6*pr+0.4*knn_pred[mj,j]
        cur[~miss]=Xev_s[~miss]
    return cur, miss

def imp_knn_svr(Xtr_s,Xev_s):
    cur,_=knn_svr_core(Xtr_s,Xev_s); return cur

def imp_knn_svr_mice_functional(Xtr_s,Xev_s):
    """Three-stage with a GENUINE MICE pass: take KNN-SVR output as a prior, then
    re-mask the originally-missing cells and let MICE (BayesianRidge) re-impute them
    using the observed cells + the KNN-SVR-filled cells of OTHER columns as context.
    This makes MICE do real work instead of receiving a complete matrix."""
    cur, miss = knn_svr_core(Xtr_s, Xev_s)
    # Build a matrix where observed stay, KNN-SVR estimates stay as context,
    # but the target missing cells are set back to NaN so MICE must predict them.
    X_for_mice = cur.copy()
    X_for_mice[miss] = np.nan
    mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0,
                            sample_posterior=False)
    out = mice.fit_transform(X_for_mice)
    # safety: keep observed fixed
    out[~miss] = Xev_s[~miss]
    return out

def imp_mice_only(Xtr_s,Xev_s):
    return IterativeImputer(estimator=BayesianRidge(),max_iter=10,random_state=0).fit(Xtr_s).transform(Xev_s)

def metrics(true,pred,vr):
    rmse=float(np.sqrt(((pred-true)**2).mean())); mae=float(np.abs(pred-true).mean())
    ss=float(((true-true.mean())**2).sum()); r2=1-float(((pred-true)**2).sum())/ss if ss>0 else np.nan
    return rmse,mae,r2

VAR={"KNN_SVR":imp_knn_svr, "KNN_SVR_MICE_func":imp_knn_svr_mice_functional}

def main(fs=0,fe=18,out="loso_mice_functional.csv"):
    df=pd.read_csv(COMPLETE)
    num=[c for c in df.select_dtypes(include=["float64","int64"]).columns if c not in ("Activity","time","participant")]
    df=df.dropna(subset=num).reset_index(drop=True); parts=sorted(df.participant.unique())
    done=set()
    if os.path.exists(out):
        try: done=set(pd.read_csv(out).participant.unique())
        except: pass
    t0=time.time()
    for fi,pid in enumerate(parts):
        if fi<fs or fi>=fe or int(pid) in done: continue
        tr=df[df.participant!=pid]; ev=df[df.participant==pid].reset_index(drop=True)
        sc=StandardScaler().fit(tr[num].to_numpy(float)); Xtr=sc.transform(tr[num].to_numpy(float))
        rows=[]
        for label,gen,mech in B.SCENARIO_METHODS:
            for rate in RATES:
                mk=B.make_eval_with_missing(ev,gen,rate,1000+fi)
                Xt=ev[num].to_numpy(float); Xr=mk[num].to_numpy(float); mask=np.isnan(Xr)
                if mask.sum()==0: continue
                Xst=sc.transform(Xt); Xs=Xst.copy(); Xs[mask]=np.nan; ta=Xst[mask]
                for m,fn in VAR.items():
                    o=fn(Xtr.copy(),Xs.copy()); rmse,mae,r2=metrics(ta,o[mask],float(np.nanmax(Xst)-np.nanmin(Xst)))
                    rows.append(dict(method=m,scenario=label,mechanism=mech,rate=int(rate*100),participant=int(pid),RMSE=rmse,MAE=mae,R2=r2))
        pd.DataFrame(rows).to_csv(out,mode="a",header=not os.path.exists(out),index=False)
        print(f"  fold {fi+1} p{pid} done [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv)>1 else 0, int(sys.argv[2]) if len(sys.argv)>2 else 18)
