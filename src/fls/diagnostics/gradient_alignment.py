"""Optional gradient/update-pressure diagnostics.

The submission-oriented checkpoint diagnostic lives in
``scripts/compute_update_pressure_diagnostics.py``. It is intentionally kept out
of the default smoke path because it performs backward passes through multiple
checkpointed runs.
"""
