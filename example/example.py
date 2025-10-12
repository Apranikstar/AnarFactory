converter = RootToPickleConverter(
    input_dir="/eos/user/h/hfatehi/yukawaBDT/off-shell-muon/",
    output_dir=".",
    tree_name="events"
)

# 1️⃣ Convert ROOT → Pickle
converter.run()

# 2️⃣ Train model
converter.train(
    signal_files=["./wzp6_ee_Hqqmunumu_ecm125.pkl", "./wzp6_ee_Hqqtaunutau_ecm125.pkl"],
    feature_list=[...],  # your full feature list
    model_output="off-shell-mu.json"
)

# 3️⃣ Run inference
converter.inference(
    signal_files=[
        "/eos/user/h/hfatehi/yukawaBDT/on-shell-muon/wzp6_ee_Hmunumuqq_ecm125.root",
        "/eos/user/h/hfatehi/yukawaBDT/on-shell-muon/wzp6_ee_Htaunutauqq_ecm125.root"
    ],
    background_files=[
        # all other ROOT files in the directory
    ],
    feature_list=[...],
    model_path="off-shell-mu.json",
    cut_value=0.5
)
