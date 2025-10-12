import warnings
import os
import glob
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


class RootToPickleConverter:
    """
    Converts ROOT files into pickled pandas DataFrames, trains XGBoost, and performs inference.
    """

    def __init__(self, input_dir, output_dir=".", tree_name="events", max_events=100_000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tree_name = tree_name
        self.max_events = max_events

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # ROOT → Pickle
    # -------------------------------------------------------------------------
    def _process_file(self, root_file: Path):
        """Process a single ROOT file and save it as a pickle."""
        print(f"\nProcessing: {root_file.name}")
        try:
            with uproot.open(root_file) as f:
                if self.tree_name not in f:
                    print(f"  [!] No tree named '{self.tree_name}', skipping.")
                    return

                tree = f[self.tree_name]
                n_entries = tree.num_entries
                print(f"  Found {n_entries:,} events.")

                n_to_read = min(n_entries, self.max_events)
                if n_entries > self.max_events:
                    print(f"  Reading first {self.max_events:,} events only.")

                df = tree.arrays(library="pd", entry_stop=n_to_read)
                df = df.copy()

                output_file = self.output_dir / f"{root_file.stem}.pkl"
                df.to_pickle(output_file)
                print(f"  Saved {len(df):,} events to {output_file}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {root_file.name}: {e}")

    def run(self):
        """Convert all ROOT files in the input directory to .pkl."""
        root_files = list(self.input_dir.glob("*.root"))
        if not root_files:
            print(f"No ROOT files found in {self.input_dir}")
            return

        for root_file in root_files:
            self._process_file(root_file)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        signal_files,
        feature_list,
        model_output="off-shell-mu.json",
        data_dir=".",
    ):
        """Train an XGBoost binary classifier on signal vs background events."""
        def load_pickle_to_df(file_list, label_value):
            dfs = []
            for fn in file_list:
                print(f"  Checking {fn} ...")
                if not os.path.exists(fn):
                    print(f"  [ERROR] Missing file: {fn}")
                    continue
                try:
                    df = pd.read_pickle(fn)
                    df["label"] = label_value
                    dfs.append(df)
                except Exception as e:
                    print(f"  [ERROR] Failed to read {fn}: {e}")
            if not dfs:
                raise RuntimeError("No valid DataFrames loaded!")
            return pd.concat(dfs, ignore_index=True)

        print("=== Collecting all pickle files ===")
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        background_files = [f for f in all_files if f not in signal_files]

        print(f"Signal files: {len(signal_files)}, Background files: {len(background_files)}")

        print("\n=== Loading SIGNAL ===")
        df_signal = load_pickle_to_df(signal_files, label_value=1)

        print("\n=== Loading BACKGROUND ===")
        df_background = load_pickle_to_df(background_files, label_value=0)

        df_all = pd.concat([df_signal, df_background], ignore_index=True)
        print(f"Total events loaded: {len(df_all):,}")

        bad_cols = [c for c in df_all.columns if not pd.api.types.is_numeric_dtype(df_all[c])]
        if bad_cols:
            for c in bad_cols:
                df_all[c] = df_all[c].apply(
                    lambda x: np.mean(x)
                    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                    else (x if isinstance(x, (int, float, np.number)) else 0)
                )
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0)

        X = df_all[feature_list].copy()
        y = df_all["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("\n=== Starting XGBoost training ===")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "eta": 0.01,
            "eval_metric": "auc",
            "tree_method": "hist",
            "nthread": 64,
        }

        evals = [(dtrain, "train"), (dtest, "test")]

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=20000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=50,
        )

        bst.save_model(model_output)
        print(f"\n✅ Training complete! Model saved as {model_output}")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def inference(
        self,
        signal_files,
        background_files,
        feature_list,
        model_path,
        tree_name="events",
        cut_value=0.5,
        chunk_size=100_000,
    ):
        """Run inference on signal and background ROOT files using a trained model."""

        def load_root_in_chunks(file_list, step=100_000):
            for arrays in uproot.iterate(
                [f"{fn}:{tree_name}" for fn in file_list],
                feature_list,
                step_size=step,
                library="ak",
            ):
                yield ak.to_dataframe(arrays)

        def apply_model_and_count(file_path, model, features, cut, step=100_000):
            total, passed = 0, 0
            all_preds = []

            for df in tqdm(load_root_in_chunks([file_path], step=step),
                           desc=f"Processing {os.path.basename(file_path)}"):
                if df.empty:
                    continue
                dmat = xgb.DMatrix(df[features])
                preds = model.predict(dmat)
                total += len(preds)
                passed += (preds > cut).sum()
                all_preds.append(preds)

            if total == 0:
                frac = 0.0
                all_preds = np.array([])
            else:
                frac = passed / total
                all_preds = np.concatenate(all_preds)

            return total, passed, frac, all_preds

        print(f"\n=== Loading model from {model_path} ===")
        bst = xgb.Booster()
        bst.load_model(model_path)
        print("✅ Model loaded successfully")

        print("\n=== Signal Samples ===")
        sig_results = {}
        for sig_file in signal_files:
            sig_total, sig_pass, sig_frac, pred_signal = apply_model_and_count(
                sig_file, bst, feature_list, cut_value, chunk_size
            )
            sig_results[os.path.basename(sig_file)] = (sig_total, sig_pass, sig_frac, pred_signal)
            print(f"{os.path.basename(sig_file)}: {sig_pass}/{sig_total} pass ({sig_frac*100:.2f}%)")

        print("\n=== Background Samples ===")
        bg_results = {}
        for bg_file in background_files:
            bg_total, bg_pass, bg_frac, pred_bg = apply_model_and_count(
                bg_file, bst, feature_list, cut_value, chunk_size
            )
            bg_results[os.path.basename(bg_file)] = (bg_total, bg_pass, bg_frac, pred_bg)
            print(f"{os.path.basename(bg_file)}: {bg_pass}/{bg_total} pass ({bg_frac*100:.2f}%)")

        pred_signal_all = np.concatenate([v[3] for v in sig_results.values()]) if sig_results else np.array([])
        pred_bg_all = np.concatenate([v[3] for v in bg_results.values()]) if bg_results else np.array([])

        print("\n✅ Inference complete.")
        print(f"  Total signal predictions: {len(pred_signal_all)}")
        print(f"  Total background predictions: {len(pred_bg_all)}")

        return sig_results, bg_results
