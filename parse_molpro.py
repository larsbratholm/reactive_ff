""" Parses molpro output files and writes extended xyz files for later use
"""

import numpy as np
import glob

def read_write_xyz(xyz_filename, exyz_filename, energy):
    """ Writes xyz files
    """
    with open(exyz_filename, "w") as f:
        with open(xyz_filename, "r") as g:
            f.write(g.readline())
            next(g)
            f.write(f"{energy}\n")
            for line in g:
                f.write(line)

def molpro_parser(filename):
    """ Parses energies from molpro output files
    """
    with open(filename) as f:
        lines = f.readlines()

    if not lines[-1].startswith(" Molpro calculation terminated"):
        print(f"error reading {filename}")
        return 0

    return float(lines[-3].split()[0])

def parse_and_write(log_datadir, output_datadir, xyz_datadir):
    """ Parse molpro output files from `log_datadir` and
        writes extended xyz to `output_datadir` using the
        standard xyz-files from `xyz_datadir`
    """
    molpro_output_files = glob.glob(log_datadir + "/*.out")
    for log_filename in molpro_output_files:
        energy = molpro_parser(log_filename)
        log_basename = log_filename.split("/")[-1].split(".")[0]
        xyz_filename = xyz_datadir + f"/{log_basename}.xyz"
        exyz_filename = output_datadir + f"/{log_basename}.xyz"
        read_write_xyz(xyz_filename, exyz_filename, energy)

if __name__ == "__main__":
    parse_and_write("./batch_dft", "./exyz", "./xyz")

