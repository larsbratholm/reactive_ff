#!/usr/bin/env bash

# Gather and unpack log_files.tar.gz
# Q-Chem log-files Used as starting
# point in normal mode sampling
cat archives/log_files/log_files?? | tar zx

# Gather and unpack xyz.tar.gz
# Contains all the generated
# structures from normal-mode sampling
cat archives/xyz/xyz?? | tar zx

# Unpack batch_xyz.tar.gz
# xyz files of the first batch of
# ~5000 structures
tar zxf archives/batch_xyz.tar.gz

# Unpack batch_dft.tar.gz
# molpro output files of cheap
# dft calculations on batch set
tar zxf archives/batch_dft.tar.gz

# Unpack batch_hybrid.tar.bz2
# molpro output files of double hybrid
# calculations on batch set
tar jxf archives/batch_hybrid.tar.bz2

# Unpack exyz_dft.tar.gz
# exyz files with cheap dft energies
tar zxf archives/exyz_dft.tar.gz

# Unpack exyz_hybrid.tar.gz
# exyz files with cheap hybrid energies
tar zxf archives/exyz_hybrid.tar.gz
