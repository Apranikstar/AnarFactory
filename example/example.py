from ../picklefactory/core import PickleFactory
import uproot
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
input_dir = "/eos/user/h/hfatehi/TestPickleFactory"
output_dir = "/eos/user/h/hfatehi/TestPickleFactory/output"
signal_name = "wzp6_ee_Henueqq_ecm125"
background_name = "wzp6_ee_qq_ecm125"
model_output = Path(output_dir) / "bdt_model.json"

# Path to one of your ROOT files to extract features
signal_file = Path(input_dir) / "wzp6_ee_Henueqq_ecm125.root"

# -----------------------------
# Load features
# -----------------------------
features = list(uproot.open(signal_file)["events"].keys())

# Remove jagged/non-numeric features that XGBoost cannot handle
#bad_features = ["Jet_nconst0", "Jet_nconst1"]  # Add more if needed
features = [f for f in features if f not in bad_features]

print(f"Using {len(features)} features for training")
print(f"Dropped features: {set(bad_features) - set(features)}")

# -----------------------------
# Initialize converter
# -----------------------------
converter = RootToPickleConverter(
    input_dir=input_dir,
    output_dir=output_dir,
    tree_name="events",
    sample_fraction=0.1,
    fcc_type="FCCee",
)

# -----------------------------
# Convert ROOT files to pickles
# -----------------------------
converter.run()

# -----------------------------
# Train XGBoost model
# -----------------------------
converter.train(
    data_dir=output_dir,
    signal_name=signal_name,
    background_name=background_name,
    feature_list=features,
    model_output=str(model_output),
)

# -----------------------------
# Run test / inference on original ROOT files
# -----------------------------
signal_root_files = [str(Path(input_dir) / "wzp6_ee_Henueqq_ecm125.root")]
background_root_files = [str(Path(input_dir) / "wzp6_ee_qq_ecm125.root")]

sig_results, bg_results = converter.inference(
    signal_files=signal_root_files,
    background_files=background_root_files,
    feature_list=features,
    model_path=str(model_output),
    tree_name="events",
    cut_value=0.988888888,
    chunk_size=100_000,
    output_dir=output_dir,
    save_predictions=True,
    output_prefix="bdt_test"
)

# -----------------------------
# Calculate expected significance
# -----------------------------
converter.calculate_significance(sig_results, bg_results)

