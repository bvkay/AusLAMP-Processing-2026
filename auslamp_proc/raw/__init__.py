"""Readers for the raw trees.

discover walks file names. edl and lemi read headers and sidecars. scan joins the two into one row per site.
None of those four opens a time-series body; cache does, and is the only module here that reads samples.

@author: ben kay (ben@auscope.org.au)
"""
