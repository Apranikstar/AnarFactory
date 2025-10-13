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


class RootToPickleConverter:
    """
    Converts ROOT files into pickled pandas DataFrames,
    trains XGBoost classifiers, performs inference, and links datasets to FCC cross sections.
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
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tree_name = tree_name
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.fcc_type = fcc_type
        self.xsec_map = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load FCC dataset parameters (cross sections)
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
