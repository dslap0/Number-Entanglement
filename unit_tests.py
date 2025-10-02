"""
@author: Nicolas Levasseur

File implementing unit tests for almost every function in the number_entanglement.py
file.
"""

import unittest

from scipy.stats import unitary_group

from number_entanglement import *


def is_degenerate(matrix):
    """
    Tests if any two eigenvalues of a given `matrix` are equal.

    @param matrix: 2D numpy array representing the matrix we want to check the
        degeneracy of.

    @return: `True` if the matrix is degenerate, `False` otherwise.
    """
    eigenvalues, _ = np.linalg.eigh(matrix)

    for i in range(len(eigenvalues) - 1):
        if np.isclose(eigenvalues[i], eigenvalues[i + 1]):
            return True

    return False


def is_hermitian(matrix, dim):
    """
    Tests if the given matrix describes a Hermitian matrix of the right dimension.

    @param matrix: 2D numpy array that should represent a Hermitian matrix of dimensions
        `dim` x `dim`.
    @param dim: Supposed dimension of the matrix.

    @return: True if `matrix` describes a Hermitian matrix of dimensions `dim` x `dim`,
        false otherwise.
    """
    right_dims = matrix.shape == (dim, dim)
    hermitian = np.allclose(matrix.conj().T, matrix)

    return right_dims and hermitian


def is_good_operator(operator, dim):
    """
    Tests if the given matrix describes a non-degenerate Hermitian matrix of the right
    dimensions.

    @param operator: 2D numpy array that should represent the matrix form of a
        non-degenerate Hermitian operator of dimensions `dim` x `dim`.
    @param dim: Supposed dimension of the matrix.

    @return: True if `matrix` describes a non-degenerate Hermitian matrix of dimensions
        `dim` x `dim`, false otherwise.
    """
    return is_hermitian(operator, dim) and not is_degenerate(operator)


def is_density_matrix(matrix, dim):
    """
    Tests if the given matrix describes a density matrix of the right dimension.

    @param matrix: 2D numpy array that should represent a density matrix of dimensions
        `dim` x `dim`.
    @param dim: Supposed dimension of the matrix.

    @return: True if `matrix` describes a density matrix of dimensions `dim` x `dim`,
        false otherwise.
    """
    hermitian = is_hermitian(matrix, dim)
    trace_condition = np.isclose(np.trace(matrix), 1.0)

    return hermitian and trace_condition


def is_pure_state(matrix, dim):
    """
    Tests if the given matrix describes a pure state density matrix of the right
    dimension.

    @param matrix: 2D numpy array that should represent a pure state density matrix of
        dimensions `dim` x `dim`.
    @param dim: Supposed dimension of the matrix.

    @return: True if `matrix` describes a density matrix of dimensions `dim` x `dim`,
        false otherwise.
    """
    density_matrix_condition = is_density_matrix(matrix, dim)
    pure_condition1 = np.allclose(matrix @ matrix, matrix)
    pure_condition2 = np.isclose(np.trace(matrix @ matrix), 1.0)

    return density_matrix_condition and pure_condition1 and pure_condition2


def is_product_of_pure_states(matrix, dim_a, dim_b):
    """
    Tests if the given matrix describes a pure state density matrix of the right
    dimensions.

    @param matrix: 2D numpy array that should represent a pure state density matrix of
        dimensions `dim` x `dim`.
    @param dim_a: Dimension of the first subsystem.
    @param dim_b: Dimension of the second subsystem.

    @return: True if `matrix` describes a density matrix of dimensions `dim` x `dim`,
        false otherwise.
    """
    dim_total = dim_a * dim_b
    density_matrix_condition = is_density_matrix(matrix, dim_total)

    _, s, _ = np.linalg.svd(matrix)

    s_reference = np.zeros(dim_total)
    s_reference[0] = 1.0

    s_condition = np.allclose(s, s_reference)

    return density_matrix_condition and s_condition


def commutator(a, b):
    return a @ b - b @ a


def is_valid_projection(states, operator, dim):
    """
    Check if the states projection is valid, meaning the projected states are still
    density matrices and the commutator between those and the `operator` is zero.

    @param states: List of 2D numpy array representing the density matrices to be
        projected.
    @param operator: 2D numpy array representing the operator we want to project unto.
    @param dim: Dimensions of the Hilbert space.

    @return: True if the projected states are density matrices that commutes with the
        `operator`, false otherwise.
    """
    projected_states = project_states(states, operator)

    for projected_state in projected_states:
        # We should also check that the state is separable, but it is NP-hard to do so.
        if not is_density_matrix(projected_state, dim):
            return False

        commut = commutator(projected_state, operator)
        if not np.allclose(commut, 0.0):
            return False

    return True


class TestNumberEntanglement(unittest.TestCase):
    def test_ket_bra(self):
        dim = 5

        vector = rng.uniform(0.0, 1.0, (dim, 2))
        vector = np.float64(vector)
        vector = vector.view(np.complex128)
        vector /= np.linalg.norm(vector)

        outer = ket_bra(vector)

        self.assertTrue(is_hermitian(outer, dim))

    def test_random_pure_state(self):
        dim = 8

        pure_state = random_pure_state(dim)

        self.assertTrue(is_pure_state(pure_state, dim))

    def test_separable_state_basis(self):
        dim_a = 2
        dim_b = 4

        dim_total = dim_a * dim_b

        basis = separable_state_basis(dim_a, dim_b)

        self.assertEqual(len(basis), dim_total * dim_total)

        for i, state in enumerate(basis):
            self.assertTrue(is_product_of_pure_states(state, dim_a, dim_b))

            for other_state in basis[:i]:
                self.assertFalse(np.allclose(state, other_state))

    def test_random_separable_state(self):
        dim_a = 2
        dim_b = 4
        dim_total = dim_a * dim_b

        basis = separable_state_basis(dim_a, dim_b)

        state = random_separable_state(basis)

        # We should also check that the state is separable, but it is NP-hard to do so.
        self.assertTrue(is_density_matrix(state, dim_total))

    def test_number_operator(self):
        dim = 3

        operator = number_operator(dim)

        zero_elements = np.count_nonzero(operator - np.diag(np.diagonal(operator)))

        self.assertEqual(zero_elements, 0)
        # Remove this test if the code allows the number operators of each subsystems
        # to be degenerate.
        self.assertTrue(is_good_operator(operator, dim))

    def test_separable_operator(self):
        # TODO
        self.assertTrue(True)

    def test_find_charge_sectors(self):
        # TODO
        self.assertTrue(True)

    def test_scramble_basis(self):
        # TODO
        self.assertTrue(True)

    def test_project_states(self):
        n_states = 5
        dim_a = 2
        dim_b = 4
        dim_total = dim_a * dim_b

        basis = separable_state_basis(dim_a, dim_b)
        states = [random_separable_state(basis) for _ in range(n_states)]

        operator = number_operator(dim_total)

        self.assertTrue(is_valid_projection(states, operator, dim_total))

        operator = separable_operator(number_operator, dim_a, dim_b)

        self.assertTrue(is_valid_projection(states, operator, dim_total))

    def test_von_neumann_entropy(self):
        dim_a = 2
        dim_b = 4
        dim_total = dim_a * dim_b

        pure_state = random_pure_state(dim_total)
        entropy = von_neumann_entropy(pure_state)
        self.assertTrue(np.isclose(entropy, 0.0))

        max_mixed_state = np.eye(dim_total) / dim_total
        entropy = von_neumann_entropy(max_mixed_state)
        self.assertTrue(np.isclose(entropy, np.log(dim_total)))

        basis = separable_state_basis(dim_a, dim_b)
        state = random_separable_state(basis)
        entropy = von_neumann_entropy(state)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, np.log(dim_total))

        unitary = unitary_group.rvs(dim_total)
        unitary_entropy = von_neumann_entropy(unitary @ state @ unitary.conj().T)
        self.assertTrue(np.isclose(entropy, unitary_entropy))

    def test_numbers_entanglement(self):
        n_states = 10
        dim_a = 2
        dim_b = 4

        basis = separable_state_basis(dim_a, dim_b)
        states = [random_separable_state(basis) for _ in range(n_states)]

        operator = separable_operator(number_operator, dim_a, dim_b)

        projected_states = project_states(states, operator)

        number_entanglement_list = numbers_entanglement(projected_states)

        for number_entanglement in number_entanglement_list:
            self.assertTrue(
                number_entanglement > 0.0 or np.isclose(number_entanglement, 0.0)
            )

        # TODO: Implement a test for symmetric separable states


if __name__ == "__main__":
    unittest.main()
