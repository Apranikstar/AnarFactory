from pathlib import Path
import uproot
import pandas as pd
import numpy as np

try:
    import cudf
    RAPIDS_AVAILABLE = True
except ImportError:
    cudf = None
    RAPIDS_AVAILABLE = False


class ROOTConverter:
    """Convert ROOT files to sampled parquet (cuDF/pandas compatible)."""

    def __init__(self, config_loader, input_dir, output_dir, tree_name="events",
                 sample_fraction=0.1, random_seed=42, use_gpu=True):
        self.config = config_loader
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tree_name = tree_name
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, root_file):
        """Read one ROOT file and save sampled parquet."""
        with uproot.open(root_file) as f:
            if self.tree_name not in f:
                print(f"  [!] No tree named '{self.tree_name}', skipping.")
                return
            tree = f[self.tree_name]
            df = tree.arrays(library="pd")

        frac = min(max(self.sample_fraction, 0), 1)
        df = df.sample(frac=frac, random_state=self.random_seed)

        xsec = self.config.get_cross_section(Path(root_file).stem)
        df["cross_section_pb"] = xsec

        out_file = self.output_dir / f"{Path(root_file).stem}.parquet"
        if self.use_gpu:
            cudf.from_pandas(df).to_parquet(out_file)
        else:
            df.to_parquet(out_file, index=False)

        print(f"✅ {root_file.name} → {out_file} (σ={xsec} pb)")

    def run(self):
        files = list(self.input_dir.glob("*.root"))
        for f in files:
            self.process_file(f)
