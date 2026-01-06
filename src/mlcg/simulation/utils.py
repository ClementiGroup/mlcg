import numpy as np
from .base import JPERKCAL, KBOLTZMANN, AVOGADRO


def calc_beta_from_temperature(temp):
    """Converts a single or a list of temperature(s) in Kelvin
    to inverse temperature(s) in mol/kcal."""
    return JPERKCAL / KBOLTZMANN / AVOGADRO / np.array(temp)
