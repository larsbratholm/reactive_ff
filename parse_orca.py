""" Parses orca output files and writes extended xyz files for later use
"""

import numpy as np
import glob
from parse_molpro import read_write_xyz

def orca_parser(filename):
    """ Parses energies from orca output files
    """
    with open(filename) as f:
        lines = f.readlines()

    if not  "ORCA TERMINATED NORMALLY" in lines[-2]:
        print(f"error reading {filename}")
        return 0

    for line in lines[::-1]:
        if line.startswith("FINAL SINGLE POINT ENERGY"):
            return float(line.split()[4])

    # should never get here
    raise SystemExit

def parse_and_write(log_datadir, output_datadir, xyz_datadir):
    """ Parse orca output files from `log_datadir` and
        writes extended xyz to `output_datadir` using the
        standard xyz-files from `xyz_datadir`
    """
    orca_output_files = glob.glob(log_datadir + "/*.log")
    for log_filename in orca_output_files:
        energy = orca_parser(log_filename)
        log_basename = log_filename.split("/")[-1].split(".")[0]
        xyz_filename = xyz_datadir + f"/{log_basename}.xyz"
        exyz_filename = output_datadir + f"/{log_basename}.xyz"
        read_write_xyz(xyz_filename, exyz_filename, energy)

if __name__ == "__main__":
    parse_and_write("./batch_hybrid", "./exyz_hybrid", "./xyz")

