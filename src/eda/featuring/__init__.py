"""
featuring sub-package — public API re-exports.
"""

from .custom_feature import (
    add_time_features,
    add_time_since_signup,
    add_transaction_velocity,
    build_feature_matrix,
)
from .preprocess import (
    apply_smote,
    encode_categoricals,
    ip_to_int,
    merge_with_geolocation,
    scale_features,
)

__all__ = [
    "add_time_features",
    "add_time_since_signup",
    "add_transaction_velocity",
    "build_feature_matrix",
    "apply_smote",
    "encode_categoricals",
    "ip_to_int",
    "merge_with_geolocation",
    "scale_features",
]
