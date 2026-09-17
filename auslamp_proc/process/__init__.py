"""Processing: the frame, the transient mask, the reference kinds, the MTH5, the Aurora pass and the EDI.

The modules are ordered as the work is: frame (signs and rotation), transients (the tail scan and the keep
mask), coherence (the estimators the choices are made on), align (member lags), references (the four kinds
and their store), selection (the one rule for which hours a pass is run on), mth5_build (the MTH5 the pass
reads), aurora_run (the pass), edi (the transfer function's header), provenance (what the run rests on), run
(the per-site CLI) and batch (one survey over many sites, one lane a site).

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

KINDS = ("single", "remote", "stack", "obs", "stack_obs")

# The word for each code key. The words are what the prose uses; the keys appear in code and in file names.
KIND_WORD = {
    "single": "single station",
    "remote": "remote site",
    "stack": "fleet stack",
    "obs": "observatory",
    "stack_obs": "stack + observatory",
}
