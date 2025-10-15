#!/usr/bin/env python3
"""
Main pipeline for HEP analysis using RAPIDS + XGBoost (GPU).
"""

import argparse
from pathlib import Path
from hep_factory import (
    PickleFactory,
    load_root_to_cudf,
    sample_cudf,
    attach_cross_section,
    train_xgboost_gpu,
    run_inference_gpu,
    calculate_significance,
)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated HEP analysis pipeline"
    )

    parser.add_argument("--input_dir", required=True, help="Directory with ROOT files")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--signal", required=True, help="Signal sample name pattern")
    parser.add_argument("--background", required=True, help="Background sample pattern")
    parser.add_argument("--features", nargs="+", required=True, help="List of features")
    parser.add_argument("--lumi", type=float, default=1e7, help="Integrated luminosity (pb^-1)")
    parser.add_argument("--cut", type=float, default=0.5, help="BDT cut threshold")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------
    # 1. Convert ROOT → cuDF Pickles
    # -------------------------------------
    pf = PickleFactory(base_dir=output_dir)
    for root_file in input_dir.glob("*.root"):
        df = load_root_to_cudf(root_file, tree_name="events")
        df = sample_cudf(df, frac=0.1)
        df = attach_cross_section(df, root_file.stem)
        pf.save(df, root_file.stem)

    # -------------------------------------
    # 2. Train GPU XGBoost
    # -------------------------------------
    model_path = output_dir / "bdt_model.json"
    bst, feature_importance = train_xgboost_gpu(
        data_dir=output_dir,
        signal_pattern=args.signal,
        background_pattern=args.background,
        features=args.features,
        model_output=model_path,
    )

    # -------------------------------------
    # 3. Inference
    # -------------------------------------
    signal_files = [f for f in output_dir.glob(f"*{args.signal}*.pkl*")]
    background_files = [f for f in output_dir.glob(f"*{args.background}*.pkl*")]

    sig_results, bg_results = run_inference_gpu(
        signal_files=signal_files,
        background_files=background_files,
        features=args.features,
        model_path=model_path,
        cut_value=args.cut,
        output_dir=output_dir,
    )

    # -------------------------------------
    # 4. Calculate Significance
    # -------------------------------------
    significance = calculate_significance(sig_results, bg_results, lumi_pb=args.lumi)
    print(f"\n✨ Final Expected Significance: {significance:.3f}\n")


if __name__ == "__main__":
    main()
