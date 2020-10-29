""" Fit fchl19 representation and forces to data from exyz files
"""

import numpy as np
import qml
from qml.utils import NUCLEAR_CHARGE, get_unique
from qml.representations import generate_fchl_acsf
from qml.kernels.gradient_kernels import get_local_kernel, get_local_symmetric_kernel, \
        get_local_gradient_kernel
from qml.math import cho_solve
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate
import time


class Data(qml.qmlearn.data.Data):
    """ The qmlearn data class on develop is a bit outdated, so
        override the xyz reader to also set energies
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of exyz files
        """

        self._set_ncompounds(len(filenames))
        self.coordinates = np.empty(self.ncompounds, dtype=object)
        self.nuclear_charges = np.empty(self.ncompounds, dtype=object)
        self.natoms = np.empty(self.ncompounds, dtype=int)
        self.energies = np.empty(self.ncompounds, dtype=float)

        for i, filename in enumerate(filenames):
            with open(filename, "r") as f:
                lines = f.readlines()

            natoms = int(lines[0])
            energy = float(lines[1])
            self.natoms[i] = natoms
            self.energies[i] = energy
            self.nuclear_charges[i] = np.empty(natoms, dtype=int)
            self.coordinates[i] = np.empty((natoms, 3), dtype=float)

            for j, line in enumerate(lines[2:natoms+2]):
                tokens = line.split()

                if len(tokens) < 4:
                    break

                self.nuclear_charges[i][j] = NUCLEAR_CHARGE[tokens[0]]
                self.coordinates[i][j] = np.asarray(tokens[1:4], dtype=float)

        # Try to convert dtype to int/float in cases where you have the
        # same molecule, just different conformers

        try:
            self.nuclear_charges = np.asarray([self.nuclear_charges[i] for i in range(self.ncompounds)], 
                    dtype=int)
            self.coordinates = np.asarray([self.coordinates[i] for i in range(self.ncompounds)],
                    dtype=float)
        except ValueError:
            pass

class FCHL_KRR(BaseEstimator):
    """ Using FCHL19 representations, Gaussian Kernels and KRR to learn energies
        with analytical gradients.
    """

    def __init__(self, data=None, 
                       representation__nRs2=24,
                       representation__nRs3=20,
                       representation__nFourier=1,
                       representation__eta2=0.32,
                       representation__eta3=2.7,
                       representation__zeta=np.pi,
                       representation__rcut=8.0,
                       representation__acut=8.0,
                       representation__two_body_decay=1.8,
                       representation__three_body_decay=0.57,
                       representation__three_body_weight=13.4,
                       kernel__sigma=10.0,
                       krr__l2_reg=1e-10):
        self.data = data
        self.elements = get_unique(self.data.nuclear_charges)
        self.representation__nRs2 = representation__nRs2
        self.representation__nRs3 = representation__nRs3
        self.representation__nFourier = representation__nFourier
        self.representation__eta2 = representation__eta2
        self.representation__eta3 = representation__eta3
        self.representation__zeta = representation__zeta
        self.representation__rcut = representation__rcut
        self.representation__acut = representation__acut
        self.representation__two_body_decay = representation__two_body_decay
        self.representation__three_body_decay = representation__three_body_decay
        self.representation__three_body_weight = representation__three_body_weight
        self.kernel__sigma = kernel__sigma
        self.krr__l2_reg = krr__l2_reg
        # Scikit-learn might parse xyz files in every cv split
        # unless the initialized data class is passed as input.
        if self.data is None:
            self.data = Data("./exyz/*.xyz")

    def fit(self, X, y=None):
        """ Fit the model. X assumed to be indices
        """
        self.training_representations, self.training_gradients = \
            self._create_representations(X)
        self.training_indices = X

        kernel = get_local_symmetric_kernel(self.training_representations,
                    data.nuclear_charges[X], self.kernel__sigma)

        kernel[np.diag_indices_from(kernel)] += self.krr__l2_reg
        self.alphas = cho_solve(kernel, data.energies[X])

    def predict(self, X, predict_forces=False):
        """ Predict energies and optionally forces of the indices X
        """
        if predict_forces:
            return self._predict_energy_and_forces(X)
        return self._predict_energy(X)

    def score(self, X, y=None):
        """ Score a set of indices X
        """
        y_pred = self.predict(X)
        y_true = self.data.energies[X]
        return - mean_absolute_error(y_true, y_pred) * 627.5

    def _predict_energy(self, indices):
        test_representations, _ = self._create_representations(indices, calculate_gradients=False)

        kernel = get_local_kernel(
                self.training_representations,
                test_representations,
                self.data.nuclear_charges[self.training_indices],
                self.data.nuclear_charges[indices],
                self.kernel__sigma)

        return np.dot(kernel, self.alphas)

    def _predict_energy_and_forces(self, indices):
        test_representations, test_gradients = self._create_representations(indices)

        energy_kernel = get_local_kernel(
                                self.training_representations,
                                test_representations,
                                self.data.nuclear_charges[self.training_indices],
                                self.data.nuclear_charges[indices],
                                self.kernel__sigma)

        force_kernel = get_local_gradient_kernel(
                                self.training_representations,
                                test_representations,
                                test_gradients,
                                self.data.nuclear_charges[self.training_indices],
                                self.data.nuclear_charges[indices],
                                self.kernel__sigma)

        kernel = np.concatenate((energy_kernel, force_kernel))
        energies_and_forces = np.dot(kernel, self.alphas)

        natoms = self.data.natoms[indices]
        n_molecules = len(indices)
        energies = energies_and_forces[:n_molecules]

        forces = []
        start_index = n_molecules
        for i in natoms:
            forces.append(energies_and_forces[start_index:start_index + i*3])
            start_index += i*3

        return energies, forces

    def _create_representations(self, indices, calculate_gradients=True):
        """ Create FCHL19 representations and gradients
        """
        nuclear_charges = self.data.nuclear_charges[indices]
        coordinates = self.data.coordinates[indices]
        natoms = data.natoms[indices]
        max_atoms = np.max(data.natoms)

        representations = []
        gradients = []
        for charge, xyz, n in zip(nuclear_charges, coordinates, natoms):
            output = generate_fchl_acsf(charge, xyz, elements=self.elements,
                            nRs2=self.representation__nRs2, nRs3=self.representation__nRs3,
                            nFourier=self.representation__nFourier, eta2=self.representation__eta2,
                            eta3=self.representation__eta3, zeta=self.representation__zeta,
                            rcut=self.representation__rcut, acut=self.representation__acut,
                            two_body_decay=self.representation__two_body_decay,
                            three_body_decay=self.representation__three_body_decay,
                            three_body_weight=self.representation__three_body_weight,
                            pad=max_atoms, gradients=calculate_gradients)

            if calculate_gradients:
                rep, grad = output
                gradients.append(grad)
            else:
                rep = output
            representations.append(rep)

        return np.asarray(representations), np.asarray(gradients)

#def _check_elements(self, nuclear_charges):
#    """
#    Check that the elements in the given nuclear_charges was
#    included in the fit.
#    """
#
#    elements_transform = get_unique(nuclear_charges)
#    if not np.isin(elements_transform, self.elements).all():
#        print("Warning: Trying to transform molecules with elements",
#              "not included during fit in the %s method." % self.__class__.__name__,
#              "%s used in training but trying to transform %s" % (str(self.elements), str(element_transform)))




if __name__ == "__main__":
    data = Data("./exyz/*.xyz")
    model = FCHL_KRR(data)
    idx = list(range(data.energies.size))
    cv = cross_validate(model, idx[:500], cv=KFold(3, shuffle=True))
    print(cv)
    #energies, forces = model.predict(idx[:5], predict_forces=True)
