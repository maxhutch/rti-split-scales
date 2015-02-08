#!/bin/bash

mpirun -n 4 ~/anaconda3/bin/python rayleigh_taylor.py
mpirun -n 4 ~/anaconda3/bin/python process.py join snapshots
mpirun -n 4 ~/anaconda3/bin/python process.py plot snapshots/*.h5

