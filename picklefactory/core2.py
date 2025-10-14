from pathlib import Path
import os
import glob
import ast
import requests
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class RootToPickleConverter:
    """
    Converts ROOT files into pickled pandas DataFrames,
    trains XGBoost classifiers, performs inference, annotates with cross sections,
    and calculates expected significance.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        input_dir,
        output_dir=".",
        tree_name="events",
        sample_fraction=0.1,
        random_seed=42,
        fcc_type=None,  # "FCCee" or "FCChh"
        lumi_pb=1e3,    # Integrated luminosity in pb^-1
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tree_name = tree_name
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.fcc_type = fcc_type
        self.xsec_map = {}
        self.lumi_pb = lumi_pb

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_fcc_parameters()

    # -------------------------------------------------------------------------
    # Load FCC Parameters (Cross Sections)
    # -------------------------------------------------------------------------
    def _load_fcc_parameters(self):
        """Load gridpacklist from param_FCCee or param_FCChh on GitHub."""
        if not self.fcc_type:
            choice = input("Is this dataset FCCee or FCChh? [FCCee/FCChh]: ").strip()
            self.fcc_type = choice

        if self.fcc_type.lower() == "fccee":
            url = "https://raw.githubusercontent.com/HEP-FCC/EventProducer/master/config/param_FCCee.py"
        elif self.fcc_type.lower() == "fcchh":
            url = "https://raw.githubusercontent.com/HEP-FCC/EventProducer/master/config/param_FCChh.py"
        else:
            print(f"[ERROR] Unknown FCC type: {self.fcc_type}")
            return

        print(f"🔄 Loading {self.fcc_type} configuration from {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text

            # Try to locate gridpacklist first; fallback to pythialist if absent
            dict_name = None
            if "gridpacklist" in content:
                dict_name = "gridpacklist"
            elif "pythialist" in content:
                dict_name = "pythialist"
            else:
                raise ValueError("No gridpacklist or pythialist found in FCC config.")

            start = content.find(dict_name)
            end = content.find("}", start)
            dict_text = content[start:end + 1].split("=", 1)[-1].strip()
            self.xsec_map = ast.literal_eval(dict_text)

            print(f"✅ Loaded {len(self.xsec_map)} entries from {self.fcc_type} ({dict_name})")

        except Exception as e:
            print(f"[ERROR] Failed to load FCC configuration: {e}")
            self.xsec_map = {}

    def get_cross_section(self, sample_name):
        """Return the cross-section (float, pb) for a given sample name."""
        for key, val in self.xsec_map.items():
            if key in sample_name:
                try:
                    return float(val[3])  # Cross-section is the 4th field
                except Exception:
                    return np.nan
        return np.nan

    # -------------------------------------------------------------------------
    # ROOT → Pickle Conversion
    # -------------------------------------------------------------------------
    def _process_file(self, root_file: Path):
        """Process a single ROOT file and save it as a sampled pickle."""
        print(f"\nProcessing: {root_file.name}")
        try:
            with uproot.open(root_file) as f:
                if self.tree_name not in f:
                    print(f"  [!] No tree named '{self.tree_name}', skipping.")
                    return

                tree = f[self.tree_name]
                n_entries = tree.num_entries
                print(f"  Found {n_entries:,} events.")

                df = tree.arrays(library="pd")
                print(f"  Loaded {len(df):,} total events.")

                frac = min(max(self.sample_fraction, 0), 1)
                df_sampled = df.sample(frac=frac, random_state=self.random_seed)
                print(f"  Selected {len(df_sampled):,} events ({frac*100:.1f}%).")

                # Attach cross section info
                xsec = self.get_cross_section(root_file.stem)
                df_sampled["cross_section_pb"] = xsec

                output_file = self.output_dir / f"{root_file.stem}.pkl"
                df_sampled.to_pickle(output_file)
                print(f"  Saved to {output_file} (σ = {xsec} pb)")

        except Exception as e:
            print(f"  [ERROR] Failed to process {root_file.name}: {e}")

    def run(self):
        """Convert all ROOT files in the input directory to .pkl format."""
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
    data_dir,
    signal_name,
    background_name,
    feature_list,
    model_output="bdt_model.json",
    params=None,
):
    """
    Train an XGBoost binary classifier.
    
    Parameters:
        data_dir (str): Directory containing pickled DataFrames.
        signal_name (str): Filename or pattern for signal files.
        background_name (str): Filename or pattern for background files.
        feature_list (list): List of features to use for training.
        model_output (str): Path to save trained model.
        params (dict): Optional XGBoost parameters.
    """
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.pkl"))

    signal_files = [f for f in all_files if signal_name in f.name]
    background_files = [f for f in all_files if background_name in f.name]

    if not signal_files:
        raise RuntimeError(f"No signal files found matching '{signal_name}'")
    if not background_files:
        raise RuntimeError(f"No background files found matching '{background_name}'")

    def load_pickle_to_df(file_list, label):
        dfs = []
        for fn in file_list:
            df = pd.read_pickle(fn)
            df["label"] = label
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    df_signal = load_pickle_to_df(signal_files, 1)
    df_background = load_pickle_to_df(background_files, 0)
    df_all = pd.concat([df_signal, df_background], ignore_index=True)

    # Ensure numeric columns
    for c in df_all.columns:
        if not pd.api.types.is_numeric_dtype(df_all[c]):
            df_all[c] = pd.to_numeric(
                df_all[c].apply(lambda x: np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) else x),
                errors="coerce"
            ).fillna(0)

    X = df_all[feature_list]
    y = df_all["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    if params is None:
        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "eta": 0.01,
            "eval_metric": "auc",
            "tree_method": "hist",
            "nthread": 64,
        }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=20000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=100,
        verbose_eval=50
    )

    bst.save_model(model_output)
    print(f"\n✅ Training complete! Model saved as {model_output}")
    # -------------------------------------------------------------------------
    # Feature Importance (Variable Ranking)
    # -------------------------------------------------------------------------
    importance = bst.get_score(importance_type='gain')

    # Convert to sorted DataFrame
    df_importance = (
    pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
    )

    print("\n📈 Variable ranking (by gain):")
    print(df_importance)

    # Optionally save to CSV or plot
    df_importance.to_csv(self.output_dir / "feature_importance.csv", index=False)
    print(f"✅ Saved variable ranking to {self.output_dir / 'feature_importance.csv'}")

    plt.figure(figsize=(20, 20),dpi=300)
    plt.barh(df_importance['feature'], df_importance['importance'])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance (gain)')
    plt.title('XGBoost Variable Importance')
    plt.tight_layout()
    plt.savefig(self.output_dir / "feature_importance.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Inference + Event-level BDT Results
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
        output_dir=".",
        save_predictions=True,
        output_prefix="bdt_results"
    ):
        """Run inference on ROOT files, save BDT outputs, return dicts of results."""

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
            all_dfs = []

            for df in tqdm(load_root_in_chunks([file_path], step=step),
                           desc=f"Processing {os.path.basename(file_path)}"):
                if df.empty:
                    continue
                dmat = xgb.DMatrix(df[features])
                preds = model.predict(dmat)
                df["bdt_score"] = preds
                df["bdt_pass"] = preds > cut

                total += len(preds)
                passed += (preds > cut).sum()
                all_preds.append(preds)
                all_dfs.append(df)

            xsec = self.get_cross_section(Path(file_path).stem)

            if total == 0:
                frac = 0.0
                df_all = pd.DataFrame()
            else:
                frac = passed / total
                df_all = pd.concat(all_dfs, ignore_index=True)
                df_all["cross_section_pb"] = xsec

            return total, passed, frac, np.concatenate(all_preds), df_all, xsec

        print(f"\n=== Loading model from {model_path} ===")
        bst = xgb.Booster()
        bst.load_model(model_path)
        print("✅ Model loaded successfully")

        results_summary = []

        # Signal
        print("\n=== Signal Samples ===")
        sig_results = {}
        for sig_file in signal_files:
            sig_total, sig_pass, sig_frac, preds, df_sig, xsec = apply_model_and_count(
                sig_file, bst, feature_list, cut_value, chunk_size
            )
            sig_results[os.path.basename(sig_file)] = (sig_total, sig_pass, sig_frac, preds, df_sig)
            print(f"{os.path.basename(sig_file)}: {sig_pass}/{sig_total} pass ({sig_frac*100:.2f}%), σ={xsec} pb")

            results_summary.append({
                "type": "signal",
                "file": os.path.basename(sig_file),
                "total": sig_total,
                "passed": sig_pass,
                "efficiency": sig_frac,
                "cross_section_pb": xsec,
            })

        # Background
        print("\n=== Background Samples ===")
        bg_results = {}
        for bg_file in background_files:
            bg_total, bg_pass, bg_frac, preds, df_bg, xsec = apply_model_and_count(
                bg_file, bst, feature_list, cut_value, chunk_size
            )
            bg_results[os.path.basename(bg_file)] = (bg_total, bg_pass, bg_frac, preds, df_bg)
            print(f"{os.path.basename(bg_file)}: {bg_pass}/{bg_total} pass ({bg_frac*100:.2f}%), σ={xsec} pb")

            results_summary.append({
                "type": "background",
                "file": os.path.basename(bg_file),
                "total": bg_total,
                "passed": bg_pass,
                "efficiency": bg_frac,
                "cross_section_pb": xsec,
            })

        # Save results
        if save_predictions:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            df_all = pd.concat(
                [v[4].assign(sample_type="signal") for v in sig_results.values()] +
                [v[4].assign(sample_type="background") for v in bg_results.values()],
                ignore_index=True
            )
            df_summary = pd.DataFrame(results_summary)

            df_all.to_pickle(output_dir / f"{output_prefix}_events.pkl")
            df_summary.to_pickle(output_dir / f"{output_prefix}_summary.pkl")

            print(f"\n✅ Saved event-level predictions to {output_dir / f'{output_prefix}_events.pkl'}")
            print(f"✅ Saved summary results to {output_dir / f'{output_prefix}_summary.pkl'}")

        print("\n✅ Inference complete.")
        return sig_results, bg_results

    # -------------------------------------------------------------------------
    # Calculate Significance
    # -------------------------------------------------------------------------
    def calculate_significance(self, sig_results, bg_results):
        """
        Calculate expected significance S/sqrt(S+B) 
        using cross sections, integrated luminosity, and BDT efficiencies.
        """
        N_signal = 0.0
        for fn, (total, passed, eff, preds, df) in sig_results.items():
            xsec = df["cross_section_pb"].iloc[0] if not df.empty else 0.0
            N_signal += xsec * self.lumi_pb * eff

        N_background = 0.0
        for fn, (total, passed, eff, preds, df) in bg_results.items():
            xsec = df["cross_section_pb"].iloc[0] if not df.empty else 0.0
            N_background += xsec * self.lumi_pb * eff

        significance = 0.0 if N_signal + N_background == 0 else N_signal / np.sqrt(N_signal + N_background)

        print(f"\n📊 Expected significance:")
        print(f"  Signal: {N_signal:.2f} events")
        print(f"  Background: {N_background:.2f} events")
        print(f"  S/sqrt(S+B) = {significance:.3f}")

        return significance
