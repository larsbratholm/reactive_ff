import numpy as np
from make_testing_input import parse_atomtypes
from make_batch_input_from_xyz import write_input_files 

def get_structure_subset(molecule_db):
    structures = []
    for key1, value1 in molecule_db.items():
        for key2, basenames in value1.items():
            mols = np.random.choice(basenames, size=min(30, len(basenames)), replace=False)
            for m in mols:
                conf = np.random.randint(31)
                structures.append(f"./xyz/{m}_{conf}.xyz")
    return structures


if __name__ == "__main__":
    molecule_db = parse_atomtypes()
    structures = get_structure_subset(molecule_db)
    write_input_files(structures)
