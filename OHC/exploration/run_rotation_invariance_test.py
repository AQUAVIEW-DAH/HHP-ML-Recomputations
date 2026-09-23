"""Is the rotation penalty information loss, or an axis-alignment artifact?
Compare raw vs full-rank rotated inputs for a tree model (axis-aligned) and a
neural network (rotation-invariant by construction). If the penalty vanishes
for the network, nothing was lost -- the axes were just made unusable."""
import numpy as np, pandas as pd, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/suramya/HHP-Prediction')
import OHC.run_locked_xgb_physics_semi_ablation as abl
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
t = [x for x in TARGETS if x.name=="tchp"][0]
cols = list(abl.FEATURE_SETS_BY_TARGET["tchp"]["global_pruned_plus_neighborhood"])
df = abl._merge_feature_tables()
w = df[pd.notna(df[t.obs_col])&pd.notna(df[t.model_col])&pd.notna(df[t.delta_col])].copy()
w = _prepare_features(w).reset_index(drop=True)
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(w[cols].apply(pd.to_numeric,errors="coerce")),columns=cols).to_numpy()
y = w[t.delta_col].to_numpy(float); yo=w[t.obs_col].to_numpy(float); ym=w[t.model_col].to_numpy(float)
ds = w["date"].dt.strftime("%Y%m%d")
fn = json.loads(abl.FOLD_PATH.read_text())
folds = _build_forward_folds(sorted(ds.unique().tolist()), n_folds=fn["n_folds"], embargo_dates=fn["embargo_dates"])
def run(make, rotate):
    oof=np.full(len(w),np.nan)
    for f in folds:
        tr=ds.isin(set(f["train_dates"])).to_numpy(); va=ds.isin(set(f["val_dates"])).to_numpy()
        sc=StandardScaler().fit(X[tr]); A,B=sc.transform(X[tr]),sc.transform(X[va])
        if rotate:
            p=PCA(n_components=X.shape[1],random_state=0).fit(A); A,B=p.transform(A),p.transform(B)
        m=make(); m.fit(A,y[tr]); oof[va]=m.predict(B)
    v=np.isfinite(oof); return float(np.abs(ym+oof-yo)[v].mean())
makers = {
 "XGBoost (axis-aligned trees)": lambda: abl._xgb_model(),
 "neural net (rotation-invariant)": lambda: MLPRegressor(hidden_layer_sizes=(64,64),max_iter=60,random_state=0,early_stopping=True),
 "ridge regression (rotation-invariant)": lambda: Ridge(alpha=1.0),
}
print(f"{'model':40s} {'raw axes':>10s} {'rotated':>10s} {'penalty':>9s}")
for name,mk in makers.items():
    a=run(mk,False); b=run(mk,True)
    print(f"{name:40s} {a:10.3f} {b:10.3f} {b-a:+9.3f}", flush=True)
