"""One site in depth: the tests, the selections with their controls, the cache variants and the forms.

The modules are ordered as workbook 05 reads them: masks (the magnetics by day, the fleet and clock tests,
the quality map, the day masks, the component masks, the best hours and the windows), centre (the shared
centre and the north-minus-east diagonal), variants (the notch and the spike screen as cache variants),
replace (a magnetic channel borrowed per channel), forms (one product per form in the run layout) and
deliver (the tipper-only product and the forms table).

The control rule every selection in this package obeys: a selection of hours or days carries a random
selection of the same size drawn without replacement from the same scored pool under a named seed, a
matched-duration contiguous selection, the selecting statistic scored kept against dropped, and the
delivered product scored against both controls.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

FORMS = ("whole", "window", "daymask", "hours", "diagonal", "notched", "despiked",
         "replace", "lender", "tipper_only")
