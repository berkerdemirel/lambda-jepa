"""Frozen probe protocols (PROTOCOL §4, D-006). `attentive_v1` arrives with token-space
extraction at M2 — vector spaces report not-applicable, never a silent fallback."""
from sslgap.probes.linear import (linear_house_v1, linear_house_v2, linear_l2_v1,  # noqa: F401
                                  linear_l2_v2, linear_raw_v1, linear_raw_v2)
from sslgap.probes.knn import knn_topk_acc, knn_self_test  # noqa: F401
