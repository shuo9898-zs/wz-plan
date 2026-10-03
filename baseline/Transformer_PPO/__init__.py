"""Plain MLP + Transformer encoder with the existing PPO V2 pipeline."""

from baseline.Transformer_PPO.encoder import (
    DEFAULT_TRANSFORMER_NETWORK_SPEC,
    PlainTransformerEncoder,
    TRANSFORMER_ENCODER_CONTRACT,
    TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS,
    TRANSFORMER_EXPECTED_POLICY_PARAMETERS,
    TRANSFORMER_FEATURES_DIM,
    TransformerNetworkSpec,
)

__all__ = [
    "DEFAULT_TRANSFORMER_NETWORK_SPEC",
    "PlainTransformerEncoder",
    "TRANSFORMER_ENCODER_CONTRACT",
    "TRANSFORMER_EXPECTED_EXTRACTOR_PARAMETERS",
    "TRANSFORMER_EXPECTED_POLICY_PARAMETERS",
    "TRANSFORMER_FEATURES_DIM",
    "TransformerNetworkSpec",
]
