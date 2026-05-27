import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.llm_config import (
    LLMConfig,
    ResearchConfig,
    FastResearchConfig,
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
        ("default", LLMConfig(), "legacy tuned large preset"),
        ("research", ResearchConfig(), "legacy research preset"),
        ("fast_research", FastResearchConfig(), "quick smoke-test preset"),
        ("5m", FiveMillionConfig(), "tiny pipeline preset"),
        ("25m", TwentyFiveMillionConfig(), "small scaling-law preset"),
        ("50m", FiftyMillionConfig(), "mid scaling-law preset"),
        ("100m", HundredMillionConfig(), "large scaling-law preset"),
    ]

    for name, config, purpose in presets:
        params = count_params(config)
        print(
            f"{name}: {params:,} params | "
            f"d_model={config.d_model}, n_layers={config.n_layers}, "
            f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, "
            f"d_ff={config.d_ff}, max_seq_len={config.max_seq_len} | "
            f"{purpose}"
        )


if __name__ == "__main__":
    main()
