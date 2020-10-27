import numpy as np

def filename_parser(filenames):
    pass

def molpro_parser(filename):
    with open(filename) as f:
        lines = f.readlines()

    if not lines[-1].startswith(" Molpro calculation terminated"):
        print(f"error reading {filename}")
        return 0

    if "f12c" in lines[-4].lower() or "DF-CCSD" in lines[-4] or "f12" not in lines[-4].lower():
        return float(lines[-3].split()[0])
    else:
        for line in lines[::-1]:
            if "!" in line and "f12a" in line.lower():
                return float(line.split()[-1])

def parse_energies(log_file_location):
    energies = np.zeros((32,20))
    for method in range(32):
        if method in [2,4]:
            continue
        for molecule in range(20):
            filename = log_file_location + f"{method}_{molecule}.out"
            try:
                energy = molpro_parser(filename)
                energies[method,molecule] = energy
            except FileNotFoundError:
                print(filename)
                pass
    return energies

def get_accuracy(energies):
    # TODO create relative energies
    # get accuracy of all methods
    #for i in range(energies.shape[0]):
    #    for j in range(energies.shape[1]):
    #        print(i, j //4, j%4, energies[i,j])
    #quit()

    relative_energies = []
    for subset in range(5):
        for pair in [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]:
            mol1 = 4*subset + pair[0]
            mol2 = 4*subset + pair[1]
            relative_energies.append(energies[:,mol1] - energies[:,mol2])

    relative_energies = np.asarray(relative_energies).T

    rmsd_errors = []
    max_errors = []
    reference_energy = relative_energies[0]
    for method in range(1,32):
        method_energies = relative_energies[method]
        sq_error = 0
        count = 0
        max_error = 0
        for i, energy in enumerate(method_energies):
            if abs(energy - reference_energy[i]) > 10:
                continue
            sq_error += (energy - reference_energy[i])**2
            max_error = max(abs(energy - reference_energy[i]), max_error)
            count += 1
        rmsd_errors.append(np.sqrt(sq_error / max(count,1e-9)))
        max_errors.append(max_error)
        print(method, rmsd_errors[method-1], max_errors[method-1])



if __name__ == "__main__":
    log_file_location = "./benchmark_logs/"
    energies = parse_energies(log_file_location)
    get_accuracy(energies)


