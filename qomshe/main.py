from qomshe.config import FCCConfigLoader
from qomshe.io_utils import ROOTConverter
from qomshe.trainer import GPUTrainer
from qomshe.inference import GPUInference
from qomshe.significance import SignificanceCalculator


def main():
    config = FCCConfigLoader("FCCee")
    config.load()

    # ROOT → Parquet
    converter = ROOTConverter(config, "input_root", "parquet_out", sample_fraction=0.05)
    converter.run()

    # Train
    trainer = GPUTrainer("parquet_out", "models")
    model = trainer.train(
        signal_name="signal",
        background_name="background",
        feature_list=["pt", "eta", "phi", "mass"],
        model_output="bdt_gpu.json"
    )

    # Inference
    infer = GPUInference(config)
    sig_res = infer.apply(["signal.root"], ["pt", "eta", "phi", "mass"], "models/bdt_gpu.json")
    bg_res = infer.apply(["background.root"], ["pt", "eta", "phi", "mass"], "models/bdt_gpu.json")

    # Significance
    calc = SignificanceCalculator(lumi_pb=1e7)
    calc.compute(sig_res, bg_res)


if __name__ == "__main__":
    main()
