# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class HybridImputerNoLeak:
    def __init__(self, knn_neighbors=5, svm_kernel="rbf", svm_C=100,
                 svm_epsilon=0.1, iterations=5, mice_max_iter=10,
                 svm_weight=0.6, random_state=42):
        self.knn_neighbors = knn_neighbors
        self.svm_kernel = svm_kernel
        self.svm_C = svm_C
        self.svm_epsilon = svm_epsilon
        self.iterations = iterations
        self.mice_max_iter = mice_max_iter
        self.svm_weight = svm_weight          # weight given to SVR vs KNN (your code: 0.6)
        self.random_state = random_state
        self.scaler_ = None
        self.svr_ = {}
        self.d_ = None

    # ---- FIT: only on complete TRAIN rows (no eval data here) ----
    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        if len(X_train) > 1500:
            X_train = X_train[np.random.RandomState(0).choice(len(X_train), 1500, replace=False)]
        self.d_ = X_train.shape[1]
        self.scaler_ = StandardScaler().fit(X_train)
        Xs = self.scaler_.transform(X_train)
        for j in range(self.d_):
            cols = [c for c in range(self.d_) if c != j]
            svr = SVR(kernel=self.svm_kernel, C=self.svm_C, epsilon=self.svm_epsilon)
            svr.fit(Xs[:, cols], Xs[:, j])
            self.svr_[j] = svr
        return self

    # ---- TRANSFORM: impute an unseen matrix with NaNs ----
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        miss = np.isnan(X)
        Xs = self.scaler_.transform(X)         # NaNs propagate
        cur = Xs.copy()

        for _ in range(self.iterations):
            # --- KNN initialization / re-estimation on observed entries ---
            # per-column KNN regression using the other columns (mean-filled) as features
            filled = cur.copy()
            col_mean = np.nanmean(cur, axis=0)
            inds = np.where(np.isnan(filled))
            filled[inds] = np.take(col_mean, inds[1])
            knn_pred = cur.copy()
            for j in range(self.d_):
                mj = miss[:, j]
                if mj.any():
                    cols = [c for c in range(self.d_) if c != j]
                    knn = KNeighborsRegressor(n_neighbors=self.knn_neighbors,
                                              metric="manhattan", weights="uniform")
                    obs = ~np.isnan(Xs[:, j])
                    knn.fit(filled[obs][:, cols], Xs[obs, j])
                    knn_pred[mj, j] = knn.predict(filled[mj][:, cols])

            # --- SVR refinement of masked positions (models pre-fit on TRAIN) ---
            svr_filled = knn_pred.copy()
            for j in range(self.d_):
                mj = miss[:, j]
                if mj.any():
                    cols = [c for c in range(self.d_) if c != j]
                    pred = self.svr_[j].predict(svr_filled[mj][:, cols])
                    # weighted blend SVR vs KNN, as in your original (0.6/0.4)
                    cur[mj, j] = self.svm_weight * pred + (1 - self.svm_weight) * knn_pred[mj, j]
            # keep observed entries fixed
            cur[~miss] = Xs[~miss]

        # --- final MICE refinement (fit on the current imputed eval matrix) ---
        mice = IterativeImputer(
            estimator=SVR(kernel=self.svm_kernel, C=self.svm_C, epsilon=self.svm_epsilon),
            max_iter=self.mice_max_iter, random_state=self.random_state)
        cur = mice.fit_transform(cur)

        return self.scaler_.inverse_transform(cur)
