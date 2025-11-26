# coding=utf-8

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

from scipy.linalg import logm
from scipy.stats import chi


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
    # Choose either
    # state_vector = rng.uniform(-1.0, 1.0, (dim, 2))
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

    basis = [np.array([])] * basis_size

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


def random_separable_state(dim_a, dim_b):
    """
    Generates a random separable state by using a convex combination of states from a
    given "basis".

    @param dim_a: Dimension of the first subsystem Hilbert space.
    @param dim_b: Dimension of the second subsystem Hilbert space.

    @return: A 2D numpy array representing the density matrix of a random separable
        state.
    """
    basis = separable_state_basis(dim_a, dim_b)

    basis_size = len(basis)

    weights = rng.uniform(0.0, 1.0, basis_size)
    weights = np.float64(weights)
    weights /= np.sum(weights)

    state = np.zeros_like(basis[0])
    for weight, density_matrix in zip(weights, basis):
        state += weight * density_matrix

    return state


# def number_operator(dim):
#     """
#     Generates a number operator of the given dimension. This number operator corresponds
#     to a site that can be occupied up to `dim - 1` times by increments of 1 each time.

#     @param dim: Dimension of the generated matrix.

#     @return: A 2D numpy array representing the matrix form of a number operator.
#     """
#     matrix = np.zeros((dim, dim))

#     for i in range(dim):
#         matrix[i][i] = i + 1

#     return matrix


def number_operator(dim):
    """
    Generates a number operator of the given dimension. This number operator corresponds
    to a system with `sqrt(dim)` qubits.

    @param dim: Dimension of the generated matrix. Should be a power of two.

    @return: A 2D numpy array representing the matrix form of a number operator.
    """
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        matrix[i][i] = int.bit_count(i)

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

    operator = np.kron(operator_a, operator_b)

    operator_a = np.kron(operator_a, np.eye(dim_b, dtype=np.float64))
    operator_b = np.kron(np.eye(dim_a, dtype=np.float64), operator_b)

    # operator = operator_a + operator_b

    return operator, operator_a, operator_b


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


def project_states(states, operator):
    """
    Projects some given `states` to all the fixed eigenvalues of a given `operator`.

    @param states: List of 2D numpy arrays representing the density matrices of the
        states to be projected.
    @param operator: 2D numpy array representing the matrix form of a non-degenerate
        Hermitian operator which is gonna be used to create the projection.

    @return: A list of 2D numpy arrays representing the projected state density matrices.
    """
    charge_sectors = find_charge_sectors(operator)

    new_states = [np.array([])] * len(states)

    for i, state in enumerate(states):
        new_state = np.zeros_like(state)
        for sector in charge_sectors.values():
            for j in sector:
                for k in sector:
                    new_state[j][k] = state[j][k]

        new_states[i] = new_state

    return new_states


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


def numbers_entanglement(states, operator):
    """
    Computes the numbers entanglement of some given `states` over a given `operator`.

    @param states: List of 2D numpy arrays representing the density matrix of a quantum
        state.
    @param operator: 2D numpy array representing an operator in the combined Hilbert
        space. This operator acts only on one of the subsystems.

    @return: A list of each state numbers entanglement for the given operator.
    """
    states_size = len(states)

    states_a = project_states(states, operator)

    number_entanglement_list = [0.0] * states_size
    for i, state, state_a in zip(range(states_size), states, states_a):
        entropy = von_neumann_entropy(state)
        entropy_a = von_neumann_entropy(state_a)

        number_entanglement_list[i] = entropy_a - entropy

    return number_entanglement_list


def numbers_entanglement_distribution(n_states, dim_a, dim_b):
    """
    Computes a distribution of number entanglements of some given dimensions.

    @param n_states: Number of states to be generated for the computation.
    @param dim_a: Dimension of the first subsystem Hilbert space.
    @param dim_b: Dimension of the second subsystem Hilbert space.

    @return: Array containing number entanglements.
    """
    states = [random_separable_state(dim_a, dim_b) for _ in range(n_states)]

    operator, operator_a, _ = separable_operator(number_operator, dim_a, dim_b)

    projected_states = project_states(states, operator)
    number_entanglement_list = numbers_entanglement(projected_states, operator_a)

    number_entanglement_array = np.array(number_entanglement_list)

    return number_entanglement_array


def histogram(x, dim):
    """
    Draws and saves an histogram plot of `x`, fitted to a chi distribution.

    @param x: Array representing the distribution data points.
    @param dim: Dimension of the combined Hilbert space.

    @return: A tuple containing the fitted parameters of the chi distribution.
    """
    plt.figure(figsize=(4, 3), dpi=100)

    bins = len(x) // 100
    count, bins_size, _ = plt.hist(
        x, bins, density=True, color="#1f77b480", label=r"$\Delta S$"
    )

    x_min = bins_size[1]
    x_max = x.max()
    y_min = 0.0
    y_max = 1.05 * np.max(count)

    mean = x.mean()
    median = np.median(x)
    plt.vlines(mean, y_min, y_max, colors="#1f77b4", linestyles="dashed")
    plt.vlines(median, y_min, y_max, colors="#1f77b4", linestyles="dotted")
    k = np.sqrt(dim)
    k, loc, scale = chi.fit(x, k, loc=0.0, scale=5e-3 / dim, method="MLE")

    x = np.linspace(x_min, x_max, 200)
    y = chi.pdf(x, k, loc, scale)
    plt.plot(
        x,
        y,
        color="#ff7f0e",
        label=rf"$\chi(x; k={k:.1f})$",
    )

    mean = chi.mean(k, loc, scale)
    median = chi.median(k, loc, scale)
    plt.vlines(mean, y_min, y_max, colors="#ff7f0e", linestyles="dashed")
    plt.vlines(median, y_min, y_max, colors="#ff7f0e", linestyles="dotted")

    plt.xlabel(r"$\Delta S$")
    plt.ylabel("Count")
    plt.ticklabel_format(axis="x", scilimits=(-1, 4))
    plt.vlines(0.0, 0.0, 0.0, colors="black", linestyles="dashed", label="Mean")
    plt.vlines(0.0, 0.0, 0.0, colors="black", linestyles="dotted", label="Median")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hist_chi_D_{dim}_qubits.png")

    return k, x.std()


def plot_params(dims, ks, stds):
    """
    Make a plot for each parameter of the chi distribution compared to the system's
    dimensions.

    @param dims: Array of int containing the dimensions of the system.
    @param ks: Array of floats containing the fitted k parameters of the chi
        distribution.
    @param locs: Array of floats containing the standard deviation of the actual distribution.
    """
    x_label = r"$d_A d_B$"

    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(dims, ks)
    plt.xlabel(x_label)
    plt.ylabel(r"k")
    plt.tight_layout()
    plt.savefig("chi_k_qubits.png")

    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(dims, stds)
    plt.xlabel(x_label)
    plt.ylabel(r"$\sigma$")
    plt.tight_layout()
    plt.savefig("chi_std_qubits.png")


if __name__ == "__main__":
    N_STATES = 10000
    dims_a = np.array(list(range(1, 4)))
    dims_a = 2**dims_a

    dims = []
    ks = []
    stds = []

    try:
        for dim_a in dims_a:
            dim_b = dim_a
            dim_total = dim_a * dim_b

            distribution = numbers_entanglement_distribution(N_STATES, dim_a, dim_b)

            k, std = histogram(distribution, dim_total)

            dims.append(dim_total)
            ks.append(k)
            stds.append(std)

    finally:
        dims = np.array(dims)
        ks = np.array(ks)
        stds = np.array(stds)

        plot_params(dims, ks, stds)
