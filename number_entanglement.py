"""
@author: Nicolas Levasseur

File calculating the mean number entanglement according to the dimension of the total
Hilbert space. The system is bipartite, and so the number entanglement is calculated
accross a wide range of separable density matrices. The number operators used to
calculate each number entanglement are composed of two non-degenerate number operators
acting on each subsystem independantly, but the combined number operator is degenerate
(in fact, it is maximally so for two non-degenerate number operators). The goal here is
to look at the upper bound of the mean number entanglement for non-degenerate number
operators acting on each subspace.
"""

from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import expm, logm
from scipy.optimize import curve_fit


SQRT_2_OVER_2 = np.sqrt(2.0) / 2.0


rng = np.random.default_rng()


def ket_bra(vector):
    """
    Computes the ket bra product of a complex vector state.

    @param vector: 1D numpy array representing the ket vector of a quantum state.

    @return: The ket bra product of a complex vector state.
    """
    return np.outer(vector, vector.conj())


def random_pure_state(dim):
    """
    Generates a random density matrix representing a pure state.

    @param dim: Dimension of the Hilbert space the pure state acts upon.

    @return: A 2D numpy array representing a pure state density matrix.
    """
    state_vector = rng.normal(0.0, SQRT_2_OVER_2, (dim, 2))
    state_vector = np.float64(state_vector)
    state_vector = state_vector.view(np.complex128)
    state_vector /= np.linalg.norm(state_vector)

    return ket_bra(state_vector)


def separable_state_basis(dim_a, dim_b):
    """
    Generates a random complete "basis" for a separable state acting on a bipartite
    Hilbert space. The term "basis" is used here in reference to a vector space basis,
    which is a set of vectors that can generate any member of a vector space by a linear
    combination. Here, the "basis" can generate any separable state acting on the
    bipartite Hilbert space by a convex combination instead of a linear one.

    @param dim_a: Dimension of the first subsystem Hilbert space.
    @param dim_b: Dimension of the second subsystem Hilbert space.

    @return: A list of `(dim_a * dim_b)**2` 2D numpy array that can generate any
        separable state acting on the bipartite Hilbert space by a convex combination.

    Notes
    -----
    This function uses Caratheodory's theorem to create a set of states that can be
    combined in a convex manner to create any separable state. We know that a separable
    state can be written as the convex combination of product states, and so that makes
    the set of separable states a convex set. The size of this set is determined by the
    fact that the separable states are a subset of Hermitian `dim_a * dim_b` matrices
    with trace 1, which gives a real dimension of `(dim_a * dim_b)**2 - 1`.
    Caratheodory's theorem states that any member of a convex set of size D can be
    written as a convex combination of at most D + 1 extremal points of the set. The
    extremal points of the set of separable states are pure product states, and so we
    need to get `(dim_a * dim_b)**2` of those to be able to generate any separable
    state.
    """
    dim_total = dim_a * dim_b
    basis_size = dim_total * dim_total

    basis = [None] * basis_size

    i = 0
    while i < basis_size:
        # There is no check here that all separable states are unique (even though it is
        # technically required), because the code is slowed down a lot by a check of
        # this kind and because the odds of this happening are astronomically low.
        pure_state_a = random_pure_state(dim_a)
        pure_state_b = random_pure_state(dim_b)
        pure_state = np.kron(pure_state_a, pure_state_b)

        basis[i] = pure_state

        i += 1

    return basis


def random_separable_state(basis):
    """
    Generates a random separable state by using a convex combination of states from a
    given "basis".

    @param basis: List of 2D numpy arrays representing the density matrices of separable
        states to be convexically combined to generate any separable state.

    @return: A 2D numpy array representing the density matrix of a random separable
        state.
    """
    basis_size = len(basis)

    weights = rng.uniform(0.0, 1.0, basis_size)
    weights = np.float64(weights)
    weights /= np.sum(weights)

    state = np.zeros_like(basis[0])
    for weight, density_matrix in zip(weights, basis):
        state += weight * density_matrix

    return state


def number_operator(dim):
    """
    Generates a number operator of the given dimension. This number operator corresponds
    to a site that can be occupied up to `dim - 1` times by increments of 1 each time.

    @param dim: Dimension of the generated matrix.

    @return: A 2D numpy array representing the matrix form of a number operator.
    """
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        matrix[i][i] = i + 1

    return matrix


def separable_operator(operator_function, dim_a, dim_b):
    """
    Generates two operators acting on two different Hilbert spaces and combines them
    into a single operator acting on both Hilbert spaces.

    @param operator_function: Function used to generate the operators that will compose
        the final one.
    @param dim_a: Dimension of the first subsystem.
    @param dim_b: Dimension of the second subsystem.

    @return: A 2D numpy array representing the matrix form of the combined operator.
    """
    operator_a = operator_function(dim_a)
    operator_b = operator_function(dim_b)

    operator = np.kron(operator_a, np.eye(dim_b, dtype=np.float64))
    operator += np.kron(np.eye(dim_a, dtype=np.float64), operator_b)

    return operator


def find_charge_sectors(operator):
    """
    Finds the charge sectors of a given diagonal `operator`.

    @param operator: A 2D array representing the matrix form of a diagonal operator.

    @return: A dictionary containing each eigenvalues of the `operator` as keys, each
        associated with an array containing every positions corresponding to that
        eigenvalue in the operator.
    """
    charge_sectors = defaultdict(list)

    for i in range(len(operator)):
        key = operator[i][i]
        charge_sectors[key].append(i)

    return charge_sectors


def scramble_basis(operator):
    """
    Finds a scrambled basis of a given diagonal `operator`. The returned basis is not
    the identity matrix if the `operator` is degenerate (it is therefore "scrambled").

    @param operator: A 2D array representing the matrix form of a diagonal operator.

    @return: A 2D array representing the matrix form of the scrambled basis.
    """
    dim = len(operator)
    charge_sectors = find_charge_sectors(operator)

    scrambled_basis = np.eye(dim, dtype=np.complex128)

    for eigenvalue in charge_sectors:
        positions = charge_sectors[eigenvalue]

        scramble_size = len(positions)

        if scramble_size == 1:
            continue

        scramble_matrix = np.zeros((dim, dim), dtype=np.complex128)

        for i, position_i in enumerate(positions):
            for position_j in positions[:i]:
                random = rng.uniform(-1.0, 1.0) + 1.0j * rng.uniform(-1.0, 1.0)
                random = np.complex128(random)

                scramble_matrix[position_i][position_j] = random
                scramble_matrix[position_j][position_i] = random.conj()

        scramble_matrix = expm(1.0j * scramble_matrix * rng.uniform(-np.pi, np.pi))

        for i in positions:
            for j in positions:
                scrambled_basis[i][j] = scramble_matrix[i][j]

    return scrambled_basis


def project_states(states, operator):
    """
    Projects some given `states` to all the fixed eigenvalues of a given `operator`.

    @param states: List of 2D numpy arrays representing the density matrices of the
        states to be projected.
    @param operator: 2D numpy array representing the matrix form of a non-degenerate
        Hermitian operator which is gonna be used to create the projection.

    @return: A list of 2D numpy arrays representing the projected state density matrices.
    """
    scrambled_basis = scramble_basis(operator)

    projection_operators = [ket_bra(vector) for vector in scrambled_basis.T]

    projected_states = [None] * len(states)
    for i, state in enumerate(states):
        projected_state = np.zeros_like(state)
        for projection_operator in projection_operators:
            projected_state += projection_operator @ state @ projection_operator

        projected_states[i] = projected_state

    return projected_states


def von_neumann_entropy(state):
    """
    Computes the von Neumann entropy of a given state.

    @param state: 2D numpy array representing the density matrix of a quantum state.

    @return: The von Neumann entropy of the state.
    """
    entropy = -np.trace(state @ logm(state, False)[0])

    if not np.isclose(np.imag(entropy), 0.0):
        raise ValueError("The von Neumann entropy is supposed to be strictly real.")

    entropy = np.real(entropy)

    return entropy


def numbers_entanglement(states):
    """
    Computes the numbers entanglement of some given `states` over a given `operator`.

    @param states: List of 2D numpy arrays representing the density matrix of a quantum
        state.

    @return: A list of each state numbers entanglement for the given operator.
    """
    states_size = len(states)

    states_a = [0.0] * states_size

    for i, state in enumerate(states):
        dim = len(state)
        state_a = np.zeros((dim, dim), dtype=np.complex128)

        for j in range(dim):
            state_a[j][j] = state[j][j]

        states_a[i] = state_a

    number_entanglement_list = [0.0] * states_size
    for i, state, state_a in zip(range(states_size), states, states_a):
        entropy = von_neumann_entropy(state)
        entropy_a = von_neumann_entropy(state_a)

        number_entanglement_list[i] = entropy_a - entropy

    return number_entanglement_list


def mean_number_entanglement(n_states, dim_a, dim_b):
    """
    Computes the mean number entanglement of some given dimensions.

    @param n_states: Number of states to be generated for the computation.
    @param dim_a: Dimension of the first subsystem Hilbert space.
    @param dim_b: Dimension of the second subsystem Hilbert space.

    @return: Tuple containing the mean number entanglement and its standard deviation.
    """
    basis = separable_state_basis(dim_a, dim_b)
    states = [random_separable_state(basis) for _ in range(n_states)]

    operator = separable_operator(number_operator, dim_a, dim_b)

    projected_states = project_states(states, operator)
    number_entanglement_list = numbers_entanglement(projected_states)

    number_entanglement_array = np.array(number_entanglement_list)

    return number_entanglement_array.mean(), number_entanglement_array.std()


def prediction(x, a, b):
    return b * x**a


def draw(x, y, y_error, y_function, x_label, y_label, title):
    """
    Draws a line graphs of the pairs `(x[i], y[i])` against a prediction
    `(x[i], y_function(x[i]))`.

    @param x: Horizontal data points.
    @param y: Vertical data points.
    @param y_error: Vertical data points error.
    @param y_function: Function used to predict the vertical data points.
    @param x_label: Horizontal axis label.
    @param x_label: Vertical axis label.
    @param title: Title of the graph's file.

    @return: Parameters of the prediction function used.
    """
    plt.figure()

    params_y, _ = curve_fit(y_function, x, y)
    a, b = params_y
    plt.scatter(x, y, c="#1f77b4", label=r"$\overline{\Delta S}$")
    plt.plot(
        x,
        y_function(x, *params_y),
        c="#1f77b4",
        label=r"${:.2f} \cdot (d_A d_B)^{{{:.2f}}}$".format(b, a),
    )

    params_y_error, _ = curve_fit(y_function, x, y_error)
    a, b = params_y_error
    plt.scatter(x, y_error, c="#ff7f0e", label=r"$\sigma(\Delta S)$")
    plt.plot(
        x,
        y_function(x, *params_y_error),
        c="#ff7f0e",
        label=r"${:.2f} \cdot (d_A d_B)^{{{:.2f}}}$".format(b, a),
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    plt.savefig(title + ".png")


if __name__ == "__main__":
    N_STATES = 1000
    dims_a = list(range(2, 40))
    dims_total = []
    means = []
    stds = []

    for dim_a in dims_a:
        dim_b = 2

        mean, std = mean_number_entanglement(N_STATES, dim_a, dim_b)

        dims_total.append(dim_a * dim_b)
        means.append(mean)
        stds.append(std)

        print(dim_a)

    dims_total = np.array(dims_total)
    means = np.array(means)
    stds = np.array(stds)

    x_label = r"$d_A d_B$"
    y_label = r"$\Delta S$"
    title = "dim_b = 2"

    draw(dims_total, means, stds, prediction, x_label, y_label, title)
