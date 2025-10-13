from pathlib import Path
import uproot
import pandas as pd
import numpy as np

class picklefactory:
    """
    Converts ROOT files into pickled pandas DataFrames,
    randomly sampling a fraction of events.
    """

    def __init__(self, input_dir, output_dir=".", tree_name="events", sample_fraction=0.1, random_seed=42):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tree_name = tree_name
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed

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

                # Read all events
                df = tree.arrays(library="pd")
                print(f"  Loaded {len(df):,} total events.")

                # Sample a fraction of the data
                frac = min(max(self.sample_fraction, 0), 1)
                df_sampled = df.sample(frac=frac, random_state=self.random_seed)
                print(f"  Selected {len(df_sampled):,} events ({frac*100:.1f}%).")

                # Save to pickle
                output_file = self.output_dir / f"{root_file.stem}.pkl"
                df_sampled.to_pickle(output_file)
                print(f"  Saved to {output_file}")

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
