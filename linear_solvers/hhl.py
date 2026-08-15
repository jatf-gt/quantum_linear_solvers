# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0.
#
# PATCHED for Qiskit 1.0+ compatibility.  Changes from original:
#   1. vector_circuit.isometry() replaced with Isometry gate via .append()
#   2. _calculate_norm rewritten to use StatevectorSampler (primitives V2)
#   3. Inner HHL class __init__ attributes promoted to outer class __init__
#      so that self._epsilon, self._num_qubits etc. are always available.

"""The HHL algorithm."""

from typing import Optional, Union, List, Callable, Tuple

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal

# PATCH 1: import Isometry gate to replace the removed .isometry() method
from qiskit.circuit.library import Isometry

from qiskit.quantum_info import Statevector

from .linear_solver import LinearSolver, LinearSolverResult
from .matrices.numpy_matrix import NumPyMatrix
from .observables.linear_system_observable import LinearSystemObservable


class HHL(LinearSolver):
    r"""HHL algorithm — patched for Qiskit 1.0+ compatibility.

    Solves the linear system A|x> = |b> using the Harrow-Hassidim-Lloyd
    algorithm with the Vázquez et al. circuit design.

    Example usage (matches the library's own docstring example)::

        import numpy as np
        from quantum_linear_solvers.linear_solvers.hhl import HHL
        from quantum_linear_solvers.linear_solvers.matrices import TridiagonalToeplitz

        matrix = TridiagonalToeplitz(2, 1, 1/3, trotter_steps=2)
        rhs    = np.array([1.0, -2.1, 3.2, -4.3])
        rhs    = rhs / np.linalg.norm(rhs)

        hhl      = HHL()
        solution = hhl.solve(matrix, rhs)
    """

    # PATCH 2: promote inner-class attributes to the outer __init__ so that
    # self._epsilon, self._scaling etc. are always defined regardless of
    # which code path is taken.
    def __init__(
        self,
        epsilon: float = 1e-2,
        sampler: Optional[object] = None,
    ) -> None:
        super().__init__()
        self._epsilon   = epsilon
        self._epsilon_r = epsilon / 3   # conditioned rotation tolerance
        self._epsilon_s = epsilon / 3   # state preparation tolerance
        self._epsilon_a = epsilon / 6   # Hamiltonian simulation tolerance
        self._scaling   = None
        self._sampler   = sampler
        self._exact_reciprocal = True
        self.scaling = 1

    # -- Properties ------------------------------------------------------------

    @property
    def scaling(self) -> float:
        """The scaling of the solution vector."""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        self._scaling = scaling

    # -- Internal helpers ------------------------------------------------------

    def _get_delta(
        self,
        n_l: int,
        lambda_min: float,
        lambda_max: float,
    ) -> float:
        """Scaling factor so that lambda_min is represented exactly on n_l bits."""
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        """
        Compute the Euclidean norm of the solution vector.

        PATCH 3: The original implementation used the Estimator primitives V1
        API which was removed in Qiskit 1.0.  We replace it with a direct
        statevector simulation using qiskit.quantum_info.Statevector, which
        is always available and does not depend on the primitives API version.

        The norm is extracted by post-selecting on the ancilla qubit being |1>
        and summing the squared amplitudes of the b-register.
        """
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size

        sv = Statevector(qc).data
        n_total = qc.num_qubits

        # Sum |amplitude|^2 for all basis states where ancilla (MSB) = 1
        # and clock register = 0 (post-QPE condition).
        # Ancilla is the last qubit = MSB of the statevector index.
        norm_sq = 0.0
        for idx in range(2**n_total):
            ancilla_bit = (idx >> (n_total - 1)) & 1
            clock_mask  = ((1 << nl) - 1) << nb
            clock_bits  = (idx & clock_mask) >> nb
            if ancilla_bit == 1 and clock_bits == 0:
                norm_sq += abs(sv[idx]) ** 2

        return float(np.sqrt(norm_sq)) / self.scaling

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        ls_observable: Optional[object] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[Callable] = None,
    ) -> Tuple:
        """
        Calculate the value of an observable on the solution state.

        Uses direct statevector simulation — no Estimator primitive needed.
        """
        nb = solution.qregs[0].size

        if ls_observable is not None:
            observable_circuit = ls_observable.observable_circuit(nb)
            post_processing    = ls_observable.post_processing
            observable_matrix  = ls_observable.observable(nb)
        else:
            observable_circuit = [QuantumCircuit(nb)]
            observable_matrix  = [np.eye(2**nb)]

        if not isinstance(observable_circuit, list):
            observable_circuit = [observable_circuit]
        if not isinstance(observable_matrix, list):
            observable_matrix = [observable_matrix]

        expectation_results = []
        for circ, obs in zip(observable_circuit, observable_matrix):
            full_circuit = solution.compose(circ)
            sv = Statevector(full_circuit).data
            n_total = full_circuit.num_qubits
            nl = solution.qregs[1].size

            # Extract b-register amplitudes post-selected on ancilla = 1.
            x_vec = np.zeros(2**nb, dtype=complex)
            for idx in range(2**n_total):
                ancilla_bit = (idx >> (n_total - 1)) & 1
                clock_mask  = ((1 << nl) - 1) << nb
                clock_bits  = (idx & clock_mask) >> nb
                if ancilla_bit == 1 and clock_bits == 0:
                    x_vec[idx & (2**nb - 1)] = sv[idx]

            exp_val = float(np.real(x_vec.conj() @ obs @ x_vec))
            expectation_results.append(exp_val)

        if len(expectation_results) == 1:
            expectation_results = expectation_results[0]

        if post_processing is not None:
            result = post_processing(expectation_results, nb, self.scaling)
        else:
            result = expectation_results

        return result, expectation_results

    # -- Main circuit construction ---------------------------------------------

    def construct_circuit(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        neg_vals: Optional[bool] = True,
    ) -> QuantumCircuit:
        """
        Construct the HHL circuit.

        Parameters
        ----------
        matrix   : system matrix A (QuantumCircuit, ndarray, or list)
        vector   : RHS vector b (QuantumCircuit, ndarray, or list)
        neg_vals : whether A has negative eigenvalues (adds sign qubit)

        Returns
        -------
        QuantumCircuit encoding the HHL solution.
        """
        # -- State preparation -------------------------------------------------
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, (list, np.ndarray)):
            vector = np.array(vector)
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)

            # PATCH 1: replace removed .isometry() with Isometry gate.
            normalised = vector / np.linalg.norm(vector)
            vector_circuit.append(
                Isometry(normalised, 0, 0),
                list(range(nb)),
            )
        else:
            raise ValueError(f"Invalid type for vector: {type(vector)}.")

        nf = 1  # number of flag qubits

        # -- Hamiltonian simulation --------------------------------------------
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, (list, np.ndarray)):
            matrix = np.array(matrix)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be Hermitian!")
            if matrix.shape[0] != 2**nb:
                raise ValueError(
                    "Input vector dimension does not match input matrix dimension! "
                    f"Vector: {nb} qubits. Matrix: {matrix.shape[0]}."
                )
            matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        else:
            raise ValueError(f"Invalid type for matrix: {type(matrix)}.")

        # Set Hamiltonian simulation tolerance.
        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        # Condition number and eigenvalue bounds.
        if (
            hasattr(matrix_circuit, "condition_bounds")
            and matrix_circuit.condition_bounds() is not None
        ):
            kappa = matrix_circuit.condition_bounds()[1]
        else:
            kappa = 1

        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        if (
            hasattr(matrix_circuit, "eigs_bounds")
            and matrix_circuit.eigs_bounds() is not None
        ):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            delta = self._get_delta(nl - neg_vals, lambda_min, lambda_max)
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            self.scaling = lambda_min
        else:
            delta = 1 / (2**nl)
            print("The solution will be calculated up to a scaling factor.")

        # -- Reciprocal circuit ------------------------------------------------
        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
            na = matrix_circuit.num_ancillas
        else:
            num_values = 2**nl
            constant   = delta
            a = int(round(num_values ** (2 / 3)))
            r = (
                2 * constant / a
                + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            )
            degree = min(
                nb,
                int(
                    np.log(
                        1
                        + (
                            16.23
                            * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                            * kappa
                            * (2 * kappa - self._epsilon_r)
                        )
                        / self._epsilon_r
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))
            breakpoints = []
            for i in range(num_intervals):
                breakpoints.append(a * (5**i))
            breakpoints.append(num_values - 1)
            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, nl
            )
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)

        # -- Assemble the full circuit -----------------------------------------
        qb = QuantumRegister(nb)   # b-register: RHS and solution
        ql = QuantumRegister(nl)   # l-register: clock / QPE eigenvalues
        qf = QuantumRegister(nf)   # flag qubit (ancilla)

        if na > 0:
            qa = AncillaRegister(na)
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        qc.append(vector_circuit, qb[:])

        # QPE
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        if na > 0:
            qc.append(
                phase_estimation,
                ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas],
            )
        else:
            qc.append(phase_estimation, ql[:] + qb[:])

        # Conditioned rotation (eigenvalue inversion)
        if self._exact_reciprocal:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(
                reciprocal_circuit.to_instruction(),
                ql[:] + [qf[0]] + qa[: reciprocal_circuit.num_ancillas],
            )

        # Inverse QPE
        if na > 0:
            qc.append(
                phase_estimation.inverse(),
                ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas],
            )
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])

        return qc

    # -- Public solve interface ------------------------------------------------

    def solve(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        observable: Optional[Union[LinearSystemObservable, List[LinearSystemObservable]]] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[Callable] = None,
    ) -> LinearSolverResult:
        """
        Solve the linear system A|x> = |b>.

        Parameters
        ----------
        matrix            : system matrix A
        vector            : RHS vector b (numpy array or QuantumCircuit)
        observable        : optional observable to evaluate on the solution
        observable_circuit: optional circuit to apply before measurement
        post_processing   : optional function applied to the raw observable value

        Returns
        -------
        LinearSolverResult with .state (QuantumCircuit) and .euclidean_norm.
        """
        if observable is not None:
            if observable_circuit is not None or post_processing is not None:
                raise ValueError(
                    "If observable is passed, observable_circuit and "
                    "post_processing cannot be set."
                )

        solution       = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self._calculate_norm(solution.state)

        if isinstance(observable, list):
            observable_all, circuit_results_all = [], []
            for obs in observable:
                obs_i, circ_i = self._calculate_observable(
                    solution.state, obs, observable_circuit, post_processing
                )
                observable_all.append(obs_i)
                circuit_results_all.append(circ_i)
            solution.observable      = observable_all
            solution.circuit_results = circuit_results_all
        elif observable is not None or observable_circuit is not None:
            solution.observable, solution.circuit_results = self._calculate_observable(
                solution.state, observable, observable_circuit, post_processing
            )

        return solution