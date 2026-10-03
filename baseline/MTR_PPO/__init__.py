"""Input-matched MTR context encoder with the existing PPO V2 pipeline."""

from baseline.MTR_PPO.encoder import (
    MTR_ENCODER_CONTRACT,
    MTR_EXPECTED_EXTRACTOR_PARAMETERS,
    MTR_FEATURES_DIM,
    InputMatchedMTREncoder,
)

__all__ = [
    "InputMatchedMTREncoder",
    "MTR_ENCODER_CONTRACT",
    "MTR_EXPECTED_EXTRACTOR_PARAMETERS",
    "MTR_FEATURES_DIM",
]
