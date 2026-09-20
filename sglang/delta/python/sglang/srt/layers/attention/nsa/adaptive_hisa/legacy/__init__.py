"""Legacy Adaptive-HISA prototypes kept for reference only.

``runtime.py`` (P-radius sealing, per-request Python loop, CPU rerank),
``sidecar.py`` (per-page atom statistics), ``cuda_select.py`` and
``csrc/adaptive_select.cu`` (launches without the current stream) are **not**
wired into production and do not implement the Phase B contract. Do not import
them from ``nsa_indexer``. ``reference.py`` is the old radius/key-energy
offline reference; it is not the production P-key builder. The pure tensor helpers
``pack_index_buffer`` / ``gather_compact`` are still used by unit tests.
"""
