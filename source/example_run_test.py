"""This is an example file showing how to run a simulation."""
"""
problem 1I Gadolinia pin (5% Gd2O3) TF=900K
moderator 600 K 
clad      600 K
fuel      900 K
moderator density 0.700 g/cc
U-235  w/o 1.8
power density 40.0 W/gU
"""

import function
import numpy as np
import pickle
import openmc_wrapper
import example_geometry
import integrator

order = 10

# Load geometry from example
geometry, volume = example_geometry.generate_geometry()
materials = example_geometry.generate_initial_number_density(order, 0.4096)
#
# Create dt vector for 5.5 months with 15 day timesteps
# dt1 = 2*24*60*60  # 15 days
#~ dt2 = 5.5*30*24*60*60  # 5.5 months
#~ dt = 2*30*24*60*60
#~ N = np.floor(dt2/dt1)
# N = 30

# dt = np.repeat([dt1], N)
dt = np.array([0.25, 6.00, 6.25, 12.50, 25.00, 50.00,                  # 100.00 EFPD 
               50.0,  50.0, 50.0, 50.0, 100.0, 100.0, 500.0, 500.0     # 1500.00 EFPD
               ])*24*60*60 

# Create settings variable
settings = openmc_wrapper.Settings()

#settings.cross_sections = "/home/jiankai/nuclear_data/hdf5_version2/nndc_hdf5/cross_sections.xml"
settings.cross_sections = "/home/jiankai/nuclear_data_old/nndc/cross_sections.xml"
#settings.chain_file     = "/home/jiankai/codes/opendeplete/chains/chain_simple.xml"
settings.chain_file     = "/home/jiankai/codes/opendeplete/chain_old_fet/chain_casl_old.xml"
#settings.openmc_call    = "/home/jiankai/codes/ClassifiedMC_Depletion/build/bin/openmc"
settings.openmc_call    = "/home/jiankai/codes/ClassifiedMC_Depletion_Tally/build/bin/openmc"
# An example for mpiexec:
# settings.openmc_call = ["mpiexec", "/home/cjosey/code/openmc/bin/openmc"]
settings.particles  = 2000
settings.batches    = 50
settings.inactive   = 10
settings.power      = 1.18989e15  # MeV/second/cm which is 190.620700286489 in unit of W/cm 
settings.dt_vec     = dt
settings.output_dir = 'fet_testing'
settings.fet_order  = order

op = function.Operator()
op.initialize(geometry, volume, materials, settings)

# Perform simulation using the MCNPX/MCNP6 algorithm
integrator.MCNPX(op)
