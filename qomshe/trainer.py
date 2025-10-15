import pandas as pd
import xgboost as xgb
import warnings
from pathlib import Path

try:
    import cudf
    from cuml.model_selection import train_test_split as cuml_train_test_split
    RAPIDS_AVAILABLE = True
except ImportError:
    cudf = None
    cuml_train_test_split = None
    RAPIDS_AVAILABLE = False


class GPUTrainer:
    """XGBoost training on GPU (cuDF) or CPU (pandas fallback)."""

    def __init__(self, data_dir, output_dir, random_seed=42, use_gpu=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not RAPIDS_AVAILABLE:
            warnings.warn("RAPIDS not found — using CPU mode.")

    def train(self, signal_name, background_name, feature_list, model_output="bdt_model.json",
              params=None, num_boost_round=2000, early_stopping_rounds=50):

        all_files = list(self.data_dir.glob("*.parquet"))
        sig_files = [f for f in all_files if signal_name in f.name]
        bkg_files = [f for f in all_files if background_name in f.name]

        def load_table(files, label):
            dfs = []
            for f in files:
                df = cudf.read_parquet(f) if self.use_gpu else pd.read_parquet(f)
                df["label"] = label
                dfs.append(df)
            return cudf.concat(dfs) if self.use_gpu else pd.concat(dfs)

        df = load_table(sig_files, 1)
        df = cudf.concat([df, load_table(bkg_files, 0)]) if self.use_gpu else pd.concat([df, load_table(bkg_files, 0)])

        X, y = df[feature_list], df["label"]
        if self.use_gpu:
            X_train, X_test, y_train, y_test = cuml_train_test_split(X, y, test_size=0.2, random_state=self.random_seed)
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_seed)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        if params is None:
            params = {
                "objective": "binary:logistic",
                "max_depth": 6,
                "eta": 0.02,
                "eval_metric": "auc",
                "tree_method": "gpu_hist" if self.use_gpu else "hist",
                "predictor": "gpu_predictor" if self.use_gpu else "auto",
            }

        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                        evals=[(dtrain, "train"), (dtest, "test")],
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=50)
        model_path = self.output_dir / model_output
        bst.save_model(model_path)
        print(f"✅ Model saved at {model_path}")

        imp = bst.get_score(importance_type="gain")
        pd.DataFrame(imp.items(), columns=["feature", "importance"]).sort_values(
            "importance", ascending=False
        ).to_csv(self.output_dir / "feature_importance.csv", index=False)

        return bst
