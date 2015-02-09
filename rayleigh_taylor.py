"""
Simulation script for 2D Rayleigh-Benard convection.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `process.py` script in this
folder can be used to merge distributed save files from parallel runs and plot
the snapshots from the command line.

To run, join, and plot using 4 processes, for instance, you could use:
$ mpiexec -n 4 python3 rayleigh_benard.py
$ mpiexec -n 4 python3 process.py join snapshots
$ mpiexec -n 4 python3 process.py plot snapshots/*.h5

On a single process, this should take ~15 minutes to run.

"""

import os
import numpy as np
import scipy as sp
from mpi4py import MPI
import time

from dedalus2 import public as de
from dedalus2.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# 2D Boussinesq hydrodynamics
problem = de.ParsedProblem(axis_names=['x','z'],
                           field_names=['p','b','u','w','bz','uz','wz'],
                           param_names=['D','N','G'])
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - D*(dx(dx(b)) + dz(bz))               = - u*dx(b) - w*bz")
problem.add_equation("dt(u) - N*(dx(dx(u)) + dz(uz)) + dx(p)       = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - N*(dx(dx(w)) + dz(wz)) + dz(p) - G*b = - u*dx(w) - w*wz")
#problem.add_left_bc("bz = 0")
#problem.add_left_bc("u = 0")
#problem.add_left_bc("w = 0")
#problem.add_right_bc("bz = 0")
#problem.add_right_bc("u = 0")
#problem.add_right_bc("w = 0", condition="(dx != 0)")
#problem.add_int_bc("p = 0", condition="(dx == 0)")
problem.add_int_bc("p = 0")
problem.add_int_bc("w = 0")
problem.add_int_bc("u = 0")

# Parameters
Lx, Lz = (0.1, 0.1)
Atwood = 1e-3
Grav   = 9.8
Visc   = 8.9e-7
Prandtl = 1.
Delta = 0.01

# Create bases and domain
x_basis = de.Fourier(1536, interval=(0, Lx), dealias=2/3)
z_basis = de.Fourier(1536, interval=(-Lz/2, Lz/2), dealias=2/3)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Finalize problem
problem.parameters['G'] = Grav 
problem.parameters['N'] = Visc
problem.parameters['D'] = Visc / Prandtl
problem.expand(domain)

# Build solver
ts = de.timesteppers.SBDF3
solver = de.solvers.IVP(problem, domain, ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-2 * np.random.standard_normal(domain.local_grid_shape) * (zt - z) * (z - zb)
b['g'] = -Atwood*sp.special.erf((z - pert)/Delta) + Atwood*sp.special.erf((z - pert + Lz/2.)/Delta) + Atwood*sp.special.erf((z - pert - Lz/2.)/Delta)
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = 10
solver.stop_wall_time = 180 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.5, max_writes=300)
snapshots.add_task("p")
snapshots.add_task("b")
snapshots.add_task("u")
snapshots.add_task("w")

# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-3, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / N", name='Re')
flow.add_property("(bz - dz(b))**2.", name='SplitErrorB')
flow.add_property("(uz - dz(u))**2.", name='SplitErrorU')
flow.add_property("(wz - dz(w))**2.", name='SplitErrorW')
flow.add_property("(dz(b))**2.", name='dBz')
flow.add_property("(dz(u))**2.", name='dUz')
flow.add_property("(dz(w))**2.", name='dWz')

 # Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re  = %f' %flow.max('Re'))
            logger.info('Split B = %e' %np.sqrt(flow.grid_average('SplitErrorB') / flow.grid_average('dBz')))
            logger.info('Split U = %e' %np.sqrt(flow.grid_average('SplitErrorU') / flow.grid_average('dUz')))
            logger.info('Split W = %e' %np.sqrt(flow.grid_average('SplitErrorW') / flow.grid_average('dWz')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %f' %(end_time-start_time))

