"""Compatibility entry point for the forward scans used by CCD analysis."""
from workbench.core.cv_io import CvScan as CcdScan, read_cv_scans


def read_ccd_scans(filepath):
    """Return each cycle's forward branch, preserving the CCD reader contract."""
    return read_cv_scans(filepath, branch="Forward")
