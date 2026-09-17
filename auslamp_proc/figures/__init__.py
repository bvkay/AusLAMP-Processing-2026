"""Figure helpers. One call per figure; the caller passes the title, the caption and the output path.

survey draws the map and the deployment register, record figure 01, site figures 02 to 05, process what each
step of workbook 03 did, transfer_functions a site's every transfer function on one page, and final the three
pages of workbook 06.

Every figure of the package is finished by figures.common.finish, which sets a short title, wraps the caption
under the axes and saves at dpi 110.

@author: ben kay (ben@auscope.org.au)
"""
