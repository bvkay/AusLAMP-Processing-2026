"""The regression that gates every port: a ported stage must reproduce the Victoria products of record.

Definition (2026-09-13): for every row of the wave's PRODUCTS_OF_RECORD table, the ported path rebuilds the product
from the same inputs (sites.csv row, windows, keep mask, reference store) and the impedance and tipper agree with the
delivered EDI to 1e-9 relative at every period, or -- where an engine's arithmetic is knowingly changed -- within the
stated tolerance with the change named in the test. A stage is 'ported' only when this passes for every wave.
Fixtures: E:/MT_Timeseries_DATA/MT_AusLAMP_GA/AusLAMP_Victoria_BK/PRODUCTS_OF_RECORD_{w2,w2m,wave3}.csv and the EDIs
they name. Not implemented yet.

@author: ben kay (ben@auscope.org.au)
"""
import pytest

@pytest.mark.skip(reason="no stage ported yet")
def test_products_of_record_reproduce():
    assert False
