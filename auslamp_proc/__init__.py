"""auslamp_proc: the AusLAMP processing method as a package.

Every function the workbooks call lives here, so a batch run and a single-site run share one code path.
No survey name is hard-coded: a survey is surveys/<name>/survey.yaml with its sites.csv and decisions.csv,
read by survey.load_survey.

@author: ben kay (ben@auscope.org.au)
"""
__version__ = "0.1.0"

from . import geo, look, observatory, register, survey  # noqa: F401
