"""An example file showing how to plot data from a simulation."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import utilities

# Installation Folder
folder = "/home/jiankai/codes/opendeplete-polynomial/opendeplete/source/"

# Casmo reference
# Note, CASMO will be more accurate due to the decay chain and the power/time
# conversion.
gd_ref = np.array([1.08156E+20, 1.05216E+20, 1.02308E+20, 9.37626E+19,
                   8.02138E+19, 6.75883E+19, 5.59204E+19, 4.52861E+19,
                   3.57487E+19, 2.73551E+19, 2.01373E+19, 1.41138E+19,
                   9.28873E+18, 5.64357E+18, 3.11343E+18, 1.54906E+18,
                   7.04083E+17, 3.00884E+17, 1.24377E+17, 5.06395E+16,
                   2.04876E+16, 8.26777E+15])
t_ref = np.array([0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
                  2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25,
                  4.50, 4.75])  # In MWd-kgHM

t_ref *= 3600 * 24 * 30 / 1.022  # Convert from MWd-kgHM to seconds

# Set variables for where the data is, and what we want to read out.
result_folder = folder + "test"
cells = [10000, 10001, 10002, 10003, 10004]
nuclides = ['Xe-135', 'U-235', 'Gd-157']

# Load data
x, y = utilities.get_atoms_volaveraged(result_folder, cells, nuclides)


plt.semilogy(x, y['Gd-157'])
plt.semilogy(t_ref, gd_ref)
plt.legend(('MCNP', 'Ref'), loc='best')
plt.show()
print(y['Gd-157'])
