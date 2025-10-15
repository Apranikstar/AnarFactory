import ast
import requests
import numpy as np


class FCCConfigLoader:
    """Fetch and parse FCC cross-section config from GitHub."""

    def __init__(self, fcc_type=None):
        self.fcc_type = fcc_type
        self.xsec_map = {}

    def load(self):
        """Load gridpacklist or pythialist."""
        if not self.fcc_type:
            choice = input("Is this dataset FCCee or FCChh? [FCCee/FCChh]: ").strip()
            self.fcc_type = choice

        if self.fcc_type.lower() == "fccee":
            url = "https://raw.githubusercontent.com/HEP-FCC/EventProducer/master/config/param_FCCee.py"
        elif self.fcc_type.lower() == "fcchh":
            url = "https://raw.githubusercontent.com/HEP-FCC/EventProducer/master/config/param_FCChh.py"
        else:
            raise ValueError(f"Unknown FCC type: {self.fcc_type}")

        print(f"🔄 Loading {self.fcc_type} configuration from {url}")
        response = requests.get(url)
        response.raise_for_status()

        content = response.text
        dict_name = "gridpacklist" if "gridpacklist" in content else (
            "pythialist" if "pythialist" in content else None
        )
        if not dict_name:
            raise ValueError("No gridpacklist or pythialist found in FCC config.")

        start = content.find(dict_name)
        end = content.find("}", start)
        dict_text = content[start:end + 1].split("=", 1)[-1].strip()
        self.xsec_map = ast.literal_eval(dict_text)
        print(f"✅ Loaded {len(self.xsec_map)} entries from {self.fcc_type} ({dict_name})")

    def get_cross_section(self, sample_name):
        for key, val in self.xsec_map.items():
            if key in sample_name:
                try:
                    return float(val[3])
                except Exception:
                    return np.nan
        return np.nan
