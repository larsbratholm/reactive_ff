""" Parse q-chem log files and created displaced structures
    using normal modes
"""


import numpy as np
import glob
import sys
import cclib
from cclib.parser import ccopen

NAME = {
    1:  'H'  ,
    2:  'He' ,
    3:  'Li' ,
    4:  'Be' ,
    5:  'B'  ,
    6:  'C'  ,
    7:  'N'  ,
    8:  'O'  ,
    9:  'F'  ,
   10:  'Ne' ,
   11:  'Na' ,
   12:  'Mg' ,
   13:  'Al' ,
   14:  'Si' ,
   15:  'P'  ,
   16:  'S'  ,
   17:  'Cl' ,
   18:  'Ar' ,
   19:  'K'  ,
   20:  'Ca' ,
   21:  'Sc' ,
   22:  'Ti' ,
   23:  'V'  ,
   24:  'Cr' ,
   25:  'Mn' ,
   26:  'Fe' ,
   27:  'Co' ,
   28:  'Ni' ,
   29:  'Cu' ,
   30:  'Zn' ,
   31:  'Ga' ,
   32:  'Ge' ,
   33:  'As' ,
   34:  'Se' ,
   35:  'Br' ,
   36:  'Kr' ,
   37:  'Rb' ,
   38:  'Sr' ,
   39:  'Y'  ,
   40:  'Zr' ,
   41:  'Nb' ,
   42:  'Mo' ,
   43:  'Tc' ,
   44:  'Ru' ,
   45:  'Rh' ,
   46:  'Pd' ,
   47:  'Ag' ,
   48:  'Cd' ,
   49:  'In' ,
   50:  'Sn' ,
   51:  'Sb' ,
   52:  'Te' ,
   53:  'I'  ,
   54:  'Xe' ,
   55:  'Cs' ,
   56:  'Ba' ,
   57:  'La' ,
   58:  'Ce' ,
   59:  'Pr' ,
   60:  'Nd' ,
   61:  'Pm' ,
   62:  'Sm' ,
   63:  'Eu' ,
   64:  'Gd' ,
   65:  'Tb' ,
   66:  'Dy' ,
   67:  'Ho' ,
   68:  'Er' ,
   69:  'Tm' ,
   70:  'Yb' ,
   71:  'Lu' ,
   72:  'Hf' ,
   73:  'Ta' ,
   74:  'W'  ,
   75:  'Re' ,
   76:  'Os' ,
   77:  'Ir' ,
   78:  'Pt' ,
   79:  'Au' ,
   80:  'Hg' ,
   81:  'Tl' ,
   82:  'Pb' ,
   83:  'Bi' ,
   84:  'Po' ,
   85:  'At' ,
   86:  'Rn' ,
   87:  'Fr' ,
   88:  'Ra' ,
   89:  'Ac' ,
   90:  'Th' ,
   91:  'Pa' ,
   92:  'U'  ,
   93:  'Np' ,
   94:  'Pu' ,
   95:  'Am' ,
   96:  'Cm' ,
   97:  'Bk' ,
   98:  'Cf' ,
   99:  'Es' ,
  100:  'Fm' ,
  101:  'Md' ,
  102:  'No' ,
  103:  'Lr' ,
  104:  'Rf' ,
  105:  'Db' ,
  106:  'Sg' ,
  107:  'Bh' ,
  108:  'Hs' ,
  109:  'Mt' ,
  110:  'Ds' ,
  111:  'Rg' ,
  112:  'Cn' ,
  114:  'Uuq',
  116:  'Uuh'}


def create_displaced_structures():
    """ Creates 10 displaced xyz files for T = 300, 600, 1200
        by normal mode sampling.
    """
    log_files = glob.glob("./log_files/**/*.log", recursive=True)
    output_folder = "./xyz"
    # Makes flat modes less likely to explode molecule
    force_min = 0.2
    for i, log_file in enumerate(log_files):
        if i < 43000:
            continue
        if i % 500 == 0:
            print(f"{i} of {len(log_files)} processed")
        mylogfile = ccopen(log_file)
        data = mylogfile.parse()
        force = get_force_constants(log_file)
        coords = data.atomcoords[-1]
        atoms = [NAME[i] for i in data.atomnos]
        modes = data.vibdisps
        # Only keep modes from last job
        modes = modes[-force.size:]
        normalized_modes = modes / np.linalg.norm(modes, axis=(1,2))[:,None,None]
        write_xyz(f"{output_folder}/{i}_0.xyz", coords, atoms)
        counter = 0
        for T in 300, 600, 1200:
            for n in range(10):
                dcoords = coords[:]
                counter += 1
                c = np.random.uniform(0, 1, size=len(force))
                c /= np.sum(c)
                r_scale = np.random.uniform(0, 1)
                c *= r_scale
                for j, mode in enumerate(normalized_modes):
                    sign = np.random.choice([-1.0, 1.0])
                    frc = max(force_min, force[j])
                    r = sign * np.sqrt(3 * c[j] * len(atoms) * 1.380e-5 * T / frc)
                    dcoords += mode * r
                write_xyz(f"{output_folder}/{i}_{counter}.xyz", dcoords, atoms)

def write_xyz(filename, coords, atoms):
    """ Writes xyz files
    """
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        for j, atom in enumerate(atoms):
            f.write("\n%2s %20.12f %20.12f %20.12f" % \
                (atom, coords[j,0], coords[j,1], coords[j,2]))
#
# atomcoords, atommasses, charge, mult, vibdisps, vibfreqs

def get_force_constants(filename):
    """ Reads the force constants of the last job in
        the Q-Chem output file
    """
    force_constants = []
    last_job_flag = False
    with open(filename) as f:
        for line in f:
            if line.startswith("Running Job"):
                tokens = line.split()
                current_job = tokens[2]
                last_job = tokens[4]
                if current_job == last_job:
                    last_job_flag = True
            if last_job_flag and line.startswith(" Force Cnst:"):
                tokens = line.split()
                force_constants.extend(tokens[2:])
            else:
                continue
    #try:
    return np.asarray(force_constants, dtype=float)
    #except:
    #    print(filename)
    #    print(force_constants)
    #    quit()

if __name__ == "__main__":
    create_displaced_structures()
