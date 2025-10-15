import os
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import xgboost as xgb
from tqdm import tqdm
from pathlib import Path

try:
    import cudf
    RAPIDS_AVAILABLE = True
except ImportError:
    cudf = None
    RAPIDS_AVAILABLE = False


class GPUInference:
    """Chunked inference on ROOT files with XGBoost."""

    def __init__(self, config_loader, use_gpu=True):
        self.config = config_loader
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE

    def _iterate_chunks(self, file_path, features, tree_name, chunk_size):
        for arrays in uproot.iterate(f"{file_path}:{tree_name}", features, step_size=chunk_size, library="ak"):
            pdf = ak.to_dataframe(arrays)
            if pdf.empty:
                continue
            yield cudf.from_pandas(pdf) if self.use_gpu else pdf

    def apply(self, files, features, model_path, tree_name="events", cut=0.5, chunk_size=100_000):
        bst = xgb.Booster()
        bst.load_model(str(model_path))

        results = {}
        for f in files:
            total, passed = 0, 0
            preds_all = []
            out_chunks = []
            xsec = self.config.get_cross_section(Path(f).stem)

            for chunk in tqdm(self._iterate_chunks(f, features, tree_name, chunk_size),
                              desc=f"Processing {os.path.basename(f)}"):
                X = chunk[features]
                preds = bst.predict(xgb.DMatrix(X))
                chunk["bdt_score"] = preds
                chunk["bdt_pass"] = preds > cut
                total += len(preds)
                passed += int((preds > cut).sum())
                preds_all.append(preds)
                out_chunks.append(chunk)

            eff = passed / total if total else 0
            df_all = cudf.concat(out_chunks) if self.use_gpu else pd.concat(out_chunks)
            df_all["cross_section_pb"] = xsec
            results[os.path.basename(f)] = (total, passed, eff, np.concatenate(preds_all), df_all)

        return results
