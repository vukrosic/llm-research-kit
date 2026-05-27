import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.llm_config import (
    FiveMillionConfig,
    TwentyFiveMillionConfig,
    FiftyMillionConfig,
    HundredMillionConfig,
)
from models.llm import MinimalLLM


def count_params(config):
    model = MinimalLLM(config)
    return sum(p.numel() for p in model.parameters())


def main():
    presets = [
        ("5M", FiveMillionConfig()),
        ("25M", TwentyFiveMillionConfig()),
        ("50M", FiftyMillionConfig()),
        ("100M", HundredMillionConfig()),
    ]

    for name, config in presets:
        params = count_params(config)
        print(
            f"{name}: {params:,} params | "
            f"d_model={config.d_model}, n_layers={config.n_layers}, "
            f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, d_ff={config.d_ff}"
        )


if __name__ == "__main__":
    main()
