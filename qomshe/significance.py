import numpy as np

class SignificanceCalculator:
    """Compute S/sqrt(S+B) with cross sections and luminosity."""

    def __init__(self, lumi_pb=1e7):
        self.lumi_pb = lumi_pb

    def compute(self, sig_results, bg_results, use_gpu=False):
        Nsig, Nbkg = 0.0, 0.0

        for _, (total, passed, eff, preds, df) in sig_results.items():
            if df is not None and len(df) > 0:
                xsec = float(df["cross_section_pb"].iloc[0])
                Nsig += xsec * self.lumi_pb * eff

        for _, (total, passed, eff, preds, df) in bg_results.items():
            if df is not None and len(df) > 0:
                xsec = float(df["cross_section_pb"].iloc[0])
                Nbkg += xsec * self.lumi_pb * eff

        sig = 0.0 if Nsig + Nbkg == 0 else Nsig / np.sqrt(Nsig + Nbkg)
        print(f"\n📊 Expected significance: S/sqrt(S+B) = {sig:.3f}")
        print(f"  Signal: {Nsig:.2f}, Background: {Nbkg:.2f}")
        return sig
