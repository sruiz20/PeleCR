# ========================================================================
#
# Imports
#
# ========================================================================
import os
import numpy.testing as npt
import pandas as pd
import unittest


# ========================================================================
#
# Test definitions
#
# ========================================================================
class ConsTestCase(unittest.TestCase):
    """Tests for conservation in Pele."""

    def test_conservation(self):
        """Are mass and energy conserved?"""

        # Load the data
        fdir = os.path.abspath(".")
        fname = os.path.join(fdir, "datlog")
        df = pd.read_csv(fname, sep="\\s+")
        init_mass = df.mass[0]
        npt.assert_allclose(df.mass, init_mass, rtol=1e-13)
        init_rho_E = df.rho_E[0]
        npt.assert_allclose(df.rho_E, init_rho_E, rtol=1e-13)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
