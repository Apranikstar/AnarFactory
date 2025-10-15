#!/usr/bin/env python3
"""
GPU-accelerated HEP analysis pipeline using RAPIDS + XGBoost (GPU).

Example usage:
    from hep_pipeline import run_pipeline

    run_pipeline(
        input_dir="data",
        output_dir="outputs",
        signal="HToBB",
        background="QCD",
        features=["pt", "eta", "phi", "mass"],
        lumi=1e7,
        cut=0.5,
    )
"""

from pathlib import Path
from qomshe import (
    PickleFactory,
    load_root_to_cudf,
    sample_cudf,
    attach_cross_section,
    train_xgboost_gpu,
    run_inference_gpu,
    calculate_significance,
)


def run_pipeline(
    input_dir,
    output_dir="outputs",
    signal=None,
    background=None,
    features=None,
    lumi=1e7,
    cut=0.5,
    sample_frac=0.1,
):
    """
    Run the full GPU-accelerated HEP analysis pipeline.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ROOT files.
    output_dir : str or Path
        Where to save intermediate and final outputs.
    signal : str
        Pattern to identify signal samples (e.g. 'HToBB').
    background : str
        Pattern to identify background samples (e.g. 'QCD').
    features : list[str]
        List of feature names used for training.
    lumi : float
        Integrated luminosity (pb^-1).
    cut : float
        BDT cut threshold.
    sample_frac : float
        Fraction of events to sample from each ROOT file (default 0.1).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------
    # 1. Convert ROOT → cuDF Pickles
    # -------------------------------------
    pf = PickleFactory(base_dir=output_dir)
    for root_file in input_dir.glob("*.root"):
        df = load_root_to_cudf(root_file, tree_name="events")
        df = sample_cudf(df, frac=sample_frac)
        df = attach_cross_section(df, root_file.stem)
        pf.save(df, root_file.stem)

    # -------------------------------------
    # 2. Train GPU XGBoost
    # -------------------------------------
    model_path = output_dir / "bdt_model.json"
    bst, feature_importance = train_xgboost_gpu(
        data_dir=output_dir,
        signal_pattern=signal,
        background_pattern=background,
        features=features,
        model_output=model_path,
    )

    # -------------------------------------
    # 3. Inference
    # -------------------------------------
    signal_files = [f for f in output_dir.glob(f"*{signal}*.pkl*")]
    background_files = [f for f in output_dir.glob(f"*{background}*.pkl*")]

    sig_results, bg_results = run_inference_gpu(
        signal_files=signal_files,
        background_files=background_files,
        features=features,
        model_path=model_path,
        cut_value=cut,
        output_dir=output_dir,
    )

    # -------------------------------------
    # 4. Calculate Significance
    # -------------------------------------
    significance = calculate_significance(sig_results, bg_results, lumi_pb=lumi)
    print(f"\n✨ Final Expected Significance: {significance:.3f}\n")

    return significance, bst, feature_importance
