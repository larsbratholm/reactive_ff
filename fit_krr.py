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
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.linear_model import Lasso
import skopt
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import numpy
import joblib

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
        self.filenames = filenames

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
                       elements=None,
                       representation_nRs2=24,
                       representation_nRs3=20,
                       representation_eta2=0.32,
                       representation_eta3=2.7,
                       representation_zeta=np.pi,
                       representation_rcut=8.0,
                       representation_acut=8.0,
                       representation_two_body_decay=1.8,
                       representation_three_body_decay=0.57,
                       representation_three_body_weight=13.4,
                       kernel_sigma=10.0,
                       krr_l2_reg=1e-10):
        self.data = data
        self.elements = elements
        if self.elements is None:
            self.elements = get_unique(self.data.nuclear_charges)
        self.representation_nRs2 = representation_nRs2
        self.representation_nRs3 = representation_nRs3
        self.representation_eta2 = representation_eta2
        self.representation_eta3 = representation_eta3
        self.representation_zeta = representation_zeta
        self.representation_rcut = representation_rcut
        self.representation_acut = representation_acut
        self.representation_two_body_decay = representation_two_body_decay
        self.representation_three_body_decay = representation_three_body_decay
        self.representation_three_body_weight = representation_three_body_weight
        self.kernel_sigma = kernel_sigma
        self.krr_l2_reg = krr_l2_reg
        # Scikit-learn might parse xyz files in every cv split
        # unless the initialized data class is passed as input.
        if self.data is None:
            self.data = Data("./exyz/*.xyz")

        self.scaler = Lasso(1e-9)

    def fit(self, X, y=None):
        """ Fit the model. X assumed to be indices
        """
        energy_offset = self._scale(X)
        self.training_representations, self.training_gradients = \
            self._create_representations(X, calculate_gradients=False)
        self.training_indices = X

        kernel = get_local_symmetric_kernel(self.training_representations,
                    data.nuclear_charges[X], self.kernel_sigma)

        kernel[np.diag_indices_from(kernel)] += self.krr_l2_reg
        energies = self.data.energies[X] - energy_offset
        self.alphas = cho_solve(kernel, energies)

    def _scale(self, indices):
        """ Fit a linear model to estimate element self-energies
        """
        energies = self.data.energies[indices]
        nuclear_charges = self.data.nuclear_charges[indices]
        coordinates = self.data.coordinates[indices]
        features = self._featurizer(nuclear_charges)
        self.scaler.fit(features, energies)
        return self.scaler.predict(features)

    def _revert_scale(self, indices):
        """ Transform predictions back to the original energy space
        """
        nuclear_charges = self.data.nuclear_charges[indices]
        coordinates = self.data.coordinates[indices]
        features = self._featurizer(nuclear_charges)
        energy_offsets = self.scaler.predict(features)
        return energy_offsets

    def _featurizer(self, nuclear_charges):
        """
        Get the counts of each element as features.
        """

        n = len(nuclear_charges)
        m = len(self.elements)
        element_to_index = {v:i for i, v in enumerate(self.elements)}
        features = np.zeros((n,m), dtype=int)

        for i, charge in enumerate(nuclear_charges):
            count_dict = {k:v for k,v in zip(*np.unique(charge, return_counts=True))}
            for key, value in count_dict.items():
                if key not in element_to_index:
                    continue
                j = element_to_index[key]
                features[i, j] = value

        return features

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
        return - mean_absolute_error(y_true, y_pred)

    def _predict_energy(self, indices):
        test_representations, _ = self._create_representations(indices, calculate_gradients=False)

        kernel = get_local_kernel(
                self.training_representations,
                test_representations,
                self.data.nuclear_charges[self.training_indices],
                self.data.nuclear_charges[indices],
                self.kernel_sigma)

        predictions = np.dot(kernel, self.alphas)
        energy_offset = self._revert_scale(indices)
        return predictions + energy_offset

    def _predict_energy_and_forces(self, indices):
        test_representations, test_gradients = self._create_representations(indices)

        energy_kernel = get_local_kernel(
                                self.training_representations,
                                test_representations,
                                self.data.nuclear_charges[self.training_indices],
                                self.data.nuclear_charges[indices],
                                self.kernel_sigma)

        force_kernel = get_local_gradient_kernel(
                                self.training_representations,
                                test_representations,
                                test_gradients,
                                self.data.nuclear_charges[self.training_indices],
                                self.data.nuclear_charges[indices],
                                self.kernel_sigma)

        kernel = np.concatenate((energy_kernel, force_kernel))
        energies_and_forces = np.dot(kernel, self.alphas)

        natoms = self.data.natoms[indices]
        n_molecules = len(indices)
        energies = energies_and_forces[:n_molecules] + self._revert_scale(indices)

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
        natoms = self.data.natoms[indices]
        max_atoms = np.max(self.data.natoms)

        representations = []
        gradients = []
        for charge, xyz, n in zip(nuclear_charges, coordinates, natoms):
            output = generate_fchl_acsf(charge, xyz, elements=self.elements,
                            nRs2=self.representation_nRs2, nRs3=self.representation_nRs3,
                            eta2=self.representation_eta2,
                            eta3=self.representation_eta3, zeta=self.representation_zeta,
                            rcut=self.representation_rcut, acut=self.representation_acut,
                            two_body_decay=self.representation_two_body_decay,
                            three_body_decay=self.representation_three_body_decay,
                            three_body_weight=self.representation_three_body_weight,
                            pad=max_atoms, gradients=calculate_gradients)

            if calculate_gradients:
                rep, grad = output
                gradients.append(grad.astype(np.float32))
            else:
                rep = output
            representations.append(rep.astype(np.float32))

        return np.asarray(representations), np.asarray(gradients)

def get_largest_errors(model):
    """ Print the biggest outlier structures
    """
    import matplotlib.pyplot as plt
    data = model.data
    scaled_energies = scaling_test(data)
    small = get_smallest_distance(data)
    large = get_largest_minimum_distance(data)
    idx = list(range(data.energies.size))
    predictions = cross_val_predict(model, idx, cv=KFold(5, shuffle=True))
    energies = data.energies[idx]
    plt.scatter(scaled_energies[idx], predictions-energies)
    plt.savefig("lol.pdf")
    errors = abs(predictions - energies)
    print(np.mean(errors))
    max_error_idx = np.argsort(-errors)
    for i in max_error_idx:
        print(data.filenames[i], errors[i], predictions[i]-energies[i], scaled_energies[i],
                small[i], large[i])

def optimize_hyper_params(model, method="gp"):
    """ Optimize hyper parameters
    """
    if method == "gp":
        return gp_optimize(model)
    elif method == "tpe":
        raise NotImplementedError
        #return tpe_optimize(model)
    print("Unknown method:", method)
    raise SystemExit

def gp_optimize(model):
    """ Optimize hyper-parameters with gaussian processes
        via the scikit-optimize library
    """
    search_space = [Integer(10, 30, prior="log-uniform", name="representation_nRs2"),
                    Integer(12, 36, prior="log-uniform", name="representation_nRs3"),
                    Real(0.64, 2, prior="log-uniform", name="representation_eta2"),
                    Real(0.7, 3, prior="log-uniform", name="representation_eta3"),
                    Real(np.pi/2, 2*np.pi, prior="log-uniform", name="representation_zeta"),
                    Real(2.0, 8.0, prior="log-uniform", name="representation_acut"),
                    Real(2.0, 10.0, prior="log-uniform", name="representation_rcut"),
                    Real(1.2, 5.0, prior="log-uniform", name="representation_two_body_decay"),
                    Real(0.57/2, 0.57*2, prior="log-uniform", name="representation_three_body_decay"),
                    Real(12, 52, prior="log-uniform", name="representation_three_body_weight"),
                    Real(9, 36.0, prior="log-uniform", name="kernel_sigma"),
                    Real(1e-8, 1e-3, prior="log-uniform", name="krr_l2_reg")
                    ]
    idx = list(range(model.data.energies.size))

    # skopt just returns the lowest error, rather than the fitted GP model,
    # so will have to share the cv-folds between model fits.
    cv = KFold(5, shuffle=True)

    @use_named_args(search_space)
    def evaluate_model(**params):
        model.set_params(**params)
        results = cross_validate(model, idx, cv=cv, n_jobs=1)
        score = results['test_score'].mean()
        score_time = results['score_time'].mean()

        return -score

    results = skopt.gp_minimize(evaluate_model, search_space, n_restarts_optimizer=20,
                                n_calls=50, verbose=True)
    print(results.x, results.fun)

def cv_prediction_dump(data, model):
    """ Prints predictions from cv splits
    """
    n = 10
    idx = list(range(data.energies.size))
    cv = KFold(n, shuffle=True)
    predictions = np.zeros((len(idx), n+1))
    for i, (train_val, test) in enumerate(cv.split(idx)):
        for j, (train, val) in enumerate(cv.split(train_val)):
            print(i, j)
            model.fit(train)
            predictions[test, j] = model.predict(test)

        model.fit(train_val)
        predictions[test, -1] = model.predict(test)

    joblib.dump(predictions, "predictions.pkl", compress=("lzma", 9), protocol=-1)

def get_cv_accuracy(model):
    cv = KFold(5, shuffle=True)
    idx = np.random.choice(range(model.data.energies.size), replace=False, size=500)
    results = cross_validate(model, idx, cv=cv, n_jobs=1)
    return results['test_score']

def save_model(model):
    nuclear_charges = model.data.nuclear_charges
    params = {'elements': model.elements,
              'nRs2': model.representation_nRs2,
              'nRs3': model.representation_nRs3,
              'eta2': model.representation_eta2,
              'eta3': model.representation_eta3,
              'zeta': model.representation_zeta,
              'rcut': model.representation_rcut,
              'acut': model.representation_acut,
              'two_body_decay': model.representation_two_body_decay,
              'three_body_decay': model.representation_three_body_decay,
              'three_body_weight': model.representation_three_body_weight,
              }
    training_representations = model.training_representations
    kernel_sigma = model.kernel_sigma
    scaler = model.scaler

    model = {'nuclear_charges': nuclear_charges,
             'params': params,
             'training_representations': training_representations,
             'sigma': kernel_sigma,
             'scaler': model.scaler
            }

    joblib.dump(model, "models/reactive_fchl.pkl", compress=("lzma", 9), protocol=-1)
    return


if __name__ == "__main__":
    np.random.seed(42)
    data = Data("./exyz_hybrid/*.xyz")
    data.energies *= 627.5
    model = FCHL_KRR(data, elements=[1,6,7,8], representation_nRs2=19,
                representation_nRs3=23, representation_eta2=1.28, representation_eta3=1.60,
                representation_zeta=3.53, representation_acut=4.78, representation_rcut=4.96,
                representation_two_body_decay=1.88, representation_three_body_decay=0.84,
                representation_three_body_weight=33.8, kernel_sigma=18, krr_l2_reg=3.1e-07)

    model.fit(list(range(data.energies.size)))
    save_model(model)
