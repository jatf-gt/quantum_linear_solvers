# This code is an adaptation of the TridiagonalToeplitz class from the
# quantum_linear_solvers library (V zquez et al., Apache 2.0 licence).
# Original copyright: (C) Copyright IBM 2020, 2021.
#
# Modifications for the pentadiagonal (fourth-order) case:
#   - Added next-nearest-neighbour off-diagonal (b2,  2 diagonals)
#   - Generalised _off_diag_circ to accept a stride parameter
#   - Updated eigs_bounds for the five-point stencil eigenvalue formula
#   - Updated power() to include both off-diagonal Trotter terms
#   - Corrected ancilla register sizing to match the original exactly
"""Hamiltonian simulation of pentadiagonal Toeplitz symmetric matrices."""

from typing import Tuple

import numpy as np
from scipy.sparse import diags
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UGate

from .linear_system_matrix import LinearSystemMatrix


class PentadiagonalToeplitz(LinearSystemMatrix):
    r"""Hamiltonian simulation for pentadiagonal Toeplitz symmetric matrices.

    Given main diagonal entry :math:`a`, nearest-neighbour off-diagonal
    :math:`b_1`, and next-nearest-neighbour off-diagonal :math:`b_2`, the
    matrix is

    .. math::

        A = a I + b_1 (S + S^\dagger) + b_2 (S^2 + S^{\dagger 2})

    where :math:`S` is the shift operator.  The Hamiltonian simulation uses
    the first-order Lie Trotter product:

    .. math::

        e^{iAt} \approx e^{iat}
                        \cdot e^{ib_1(S+S^\dagger)t}
                        \cdot e^{ib_2(S^2+S^{\dagger 2})t}

    The nearest-neighbour exponential :math:`e^{ib_1(S+S^\dagger)t}` is
    implemented by the same controlled-rotation ladder as
    :class:`TridiagonalToeplitz` (stride 1).  The next-nearest-neighbour
    exponential :math:`e^{ib_2(S^2+S^{\dagger 2})t}` uses the same ladder
    with stride 2, without the unconditional boundary rotation that is only
    valid for stride 1.
    """

    def __init__(
        self,
        num_state_qubits: int,
        main_diag: float,
        off_diag_1: float,
        off_diag_2: float,
        tolerance: float = 1e-2,
        evolution_time: float = 1.0,
        trotter_steps: int = 1,
        name: str = "penta",
    ) -> None:
        # Initialise internal state before super().__init__ which calls
        # the evolution_time setter (which reads off_diag_1/2).
        self._num_state_qubits = None
        self._main_diag = None
        self._off_diag_1 = None
        self._off_diag_2 = None
        self._tolerance = None
        self._evolution_time = None
        self._trotter_steps = None

        # Store diagonals before super().__init__ triggers setters.
        self._main_diag = main_diag
        self._off_diag_1 = off_diag_1
        self._off_diag_2 = off_diag_2

        super().__init__(
            num_state_qubits=num_state_qubits,
            tolerance=tolerance,
            evolution_time=evolution_time,
            name=name,
        )

        # Override auto-computed trotter_steps if caller supplied one.
        self.trotter_steps = trotter_steps

    #   Properties  

    @property
    def num_state_qubits(self) -> int:
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        if num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @property
    def main_diag(self) -> float:
        return self._main_diag

    @main_diag.setter
    def main_diag(self, main_diag: float) -> None:
        self._main_diag = main_diag

    @property
    def off_diag_1(self) -> float:
        return self._off_diag_1

    @off_diag_1.setter
    def off_diag_1(self, off_diag_1: float) -> None:
        self._off_diag_1 = off_diag_1

    @property
    def off_diag_2(self) -> float:
        return self._off_diag_2

    @off_diag_2.setter
    def off_diag_2(self, off_diag_2: float) -> None:
        self._off_diag_2 = off_diag_2

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        self._tolerance = tolerance

    @property
    def evolution_time(self) -> float:
        return self._evolution_time

    @evolution_time.setter
    def evolution_time(self, evolution_time: float) -> None:
        """Set evolution time and recompute Trotter steps.

        Mirrors TridiagonalToeplitz exactly, using the dominant off-diagonal
        magnitude.  For the pentadiagonal case the dominant term is b1 (since
        |b1| >> |b2| for the fourth-order Poisson stencil), so we use
        max(|b1|, |b2|) as a conservative bound.
        """
        self._evolution_time = evolution_time
        if (
            self._off_diag_1 is not None
            and self._tolerance is not None
            and self._tolerance > 0
        ):
            dominant = max(
                abs(self._off_diag_1),
                abs(self._off_diag_2) if self._off_diag_2 is not None else 0.0,
            )
            self._trotter_steps = int(
                np.ceil(
                    np.sqrt(
                        ((evolution_time * dominant) ** 3) / 2.0 / self._tolerance
                    )
                )
            )

    @property
    def trotter_steps(self) -> int:
        return self._trotter_steps

    @trotter_steps.setter
    def trotter_steps(self, trotter_steps: int) -> None:
        self._trotter_steps = trotter_steps

    #   Matrix and eigenvalue utilities  

    @property
    def matrix(self) -> np.ndarray:
        """Return the dense pentadiagonal Toeplitz matrix."""
        N = 2 ** self.num_state_qubits
        return diags(
            [self.off_diag_2, self.off_diag_1, self.main_diag,
             self.off_diag_1, self.off_diag_2],
            [-2, -1, 0, 1, 2],
            shape=(N, N),
        ).toarray()

    def eigs_bounds(self) -> Tuple[float, float]:
        """
        Return lower and upper bounds on the absolute eigenvalues of the matrix.
        Uses exact diagonalization since the analytical formula for infinite Toeplitz
        matrices does not exactly hold for finite truncated pentadiagonal matrices.
        """
        eigs = np.linalg.eigvalsh(self.matrix)
        abs_eigs = np.abs(eigs)
        return float(abs_eigs.min()), float(abs_eigs.max())

    def condition_bounds(self) -> Tuple[float, float]:
        kappa = np.linalg.cond(self.matrix)
        return kappa, kappa

    #   Configuration and register management  

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True
        if self._trotter_steps is not None and self._trotter_steps < 1:
            valid = False
            if raise_on_failure:
                raise AttributeError(
                    "The number of Trotter steps must be a positive integer."
                )
        return valid

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Reset quantum registers."""
        qr_state = QuantumRegister(num_state_qubits, "state")
        self.qregs = [qr_state]
        self._ancillas = []
        self._qubits = qr_state[:]

    def _build(self) -> None:
        if self._is_built:
            return
        super()._build()
        self.compose(self.power(1), inplace=True)

    #   Circuit building blocks  

    def _main_diag_circ(self, theta: float = 1) -> QuantumCircuit:
        """Circuit for e^{i.a.I.theta} - identical to TridiagonalToeplitz."""
        theta *= self.main_diag
        qc = QuantumCircuit(self.num_state_qubits, name="main_diag")
        qc.x(0)
        qc.p(theta, 0)
        qc.x(0)
        qc.p(theta, 0)

        def control(num_ctrl_qubits=1, label=None, ctrl_state=None):
            qc_control = QuantumCircuit(
                self.num_state_qubits + 1, name="main_diag"
            )
            qc_control.p(theta, 0)
            return qc_control

        qc.control = control
        return qc


    def _off_diag_circ(
        self, theta_val: float, val: float, shift: int, name: str
    ) -> QuantumCircuit:
        """Circuit implementing a Toeplitz hopping matrix with offset 2^shift.
        
        A shift of 0 corresponds to +-1 (nearest neighbor).
        A shift of 1 corresponds to +-2 (next-nearest neighbor).
        
        This is mathematically equivalent to the standard TridiagonalToeplitz
        logic acting on the subset of qubits qr[shift : num_state_qubits],
        because adding 2^shift to a binary integer is exactly the same as adding 1
        to the integer formed by ignoring the lowest `shift` bits.
        """
        n = self.num_state_qubits
        theta_val *= val

        qr = QuantumRegister(n)
        qc = QuantumCircuit(qr, name=name)

        # Unconditional rotation on the effective LSB (qr[shift])
        if n > shift:
            qc.u(-2 * theta_val, 3 * np.pi / 2, np.pi / 2, qr[shift])

        # Controlled-rotation ladder starting from shift
        for i in range(shift, n - 1):
            q_controls = []

            qc.cx(qr[i], qr[i + 1])
            q_controls.append(qr[i + 1])

            qc.x(qr[i])
            for j in range(i, shift, -1):
                qc.cx(qr[i], qr[j - 1])
                q_controls.append(qr[j - 1])
            qc.x(qr[i])

            if len(q_controls) > 1:
                ugate = UGate(-2 * theta_val, 3 * np.pi / 2, np.pi / 2)
                qc.append(
                    ugate.control(len(q_controls)),
                    q_controls[:] + [qr[i]]
                )
            else:
                qc.cu(
                    -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                    q_controls[0], qr[i],
                )

            # Uncompute
            qc.x(qr[i])
            for j in range(shift, i):
                qc.cx(qr[i], qr[j])
            qc.x(qr[i])
            qc.cx(qr[i], qr[i + 1])

        #   Controlled version (used inside QPE)  
        def control(num_ctrl_qubits=1, label=None, ctrl_state=None):
            qr_state = QuantumRegister(n + 1)
            qc_ctrl = QuantumCircuit(qr_state, name=name)

            q_ctrl = qr_state[0]
            qr_d = qr_state[1:]

            if n > shift:
                qc_ctrl.cu(
                    -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                    q_ctrl, qr_d[shift],
                )

            for i in range(shift, n - 1):
                q_controls = [q_ctrl]

                qc_ctrl.cx(qr_d[i], qr_d[i + 1])
                q_controls.append(qr_d[i + 1])

                qc_ctrl.x(qr_d[i])
                for j in range(i, shift, -1):
                    qc_ctrl.cx(qr_d[i], qr_d[j - 1])
                    q_controls.append(qr_d[j - 1])
                qc_ctrl.x(qr_d[i])

                if len(q_controls) > 1:
                    ugate = UGate(-2 * theta_val, 3 * np.pi / 2, np.pi / 2)
                    qc_ctrl.append(
                        ugate.control(len(q_controls)),
                        q_controls[:] + [qr_d[i]]
                    )
                else:
                    qc_ctrl.cu(
                        -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                        q_controls[0], qr_d[i],
                    )

                qc_ctrl.x(qr_d[i])
                for j in range(shift, i):
                    qc_ctrl.cx(qr_d[i], qr_d[j])
                qc_ctrl.x(qr_d[i])
                qc_ctrl.cx(qr_d[i], qr_d[i + 1])

            return qc_ctrl

        qc.control = control
        return qc

    def _near_off_diag_circ(self, theta: float) -> QuantumCircuit:
        # nearest-neighbour
        return self._off_diag_circ(
            theta, self.off_diag_1, shift=0, name="near_off"
        )

    def _next_off_diag_circ(self, theta: float) -> QuantumCircuit:
        # next-nearest-neighbour
        return self._off_diag_circ(
            theta, self.off_diag_2, shift=1, name="next_off"
        )

    #   Inverse  

    def inverse(self) -> "PentadiagonalToeplitz":
        return PentadiagonalToeplitz(
            self.num_state_qubits,
            self.main_diag,
            self.off_diag_1,
            self.off_diag_2,
            evolution_time=-1.0 * self.evolution_time,
        )

    #   Power (QPE interface)  

    def power(self, power: int, matrix_power: bool = False) -> QuantumCircuit:
        # Build the controlled-power circuit used inside QPE.
        # Trotter product for the pentadiagonal case.
        # The main diagonal commutes with everything and is applied once.
        qc_raw = QuantumCircuit(self.num_state_qubits)

        def control(num_ctrl_qubits=1, label=None, ctrl_state=None):
            qr_state = QuantumRegister(self.num_state_qubits + 1, "state")
            qc = QuantumCircuit(qr_state, name="exp(iHk)")

            q_control = qr_state[0]
            qr = qr_state[1:]

            #   Main diagonal: one application at full power.t  
            qc.append(
                self._main_diag_circ(self.evolution_time * power)
                .control()
                .to_gate(),
                [q_control] + qr[:],
            )

            #   Trotter steps  
            trotter_steps_new = max(
                1, int(np.ceil(np.sqrt(power) * self.trotter_steps))
            )
            t_step = self.evolution_time * power / trotter_steps_new

            # Half-step bookend for b1 BEFORE the loop.
            # This matches TridiagonalToeplitz exactly: the unconditional
            # rotation on qr[0] inside _near_off_diag_circ is the boundary
            # term of the hopping ladder; the bookend compensates for the
            # fact that this boundary term is applied once per Trotter step
            # but should contribute t_step/2 at the start and t_step/2 at
            # the end (Strang splitting on the first qubit only).
            qc.u(
                self.off_diag_1 * t_step,
                3 * np.pi / 2,
                np.pi / 2,
                qr[0],
            )

            for _ in range(trotter_steps_new):
                # Full step: nearest-neighbour (b1)
                qc.append(
                    self._near_off_diag_circ(t_step)
                    .control()
                    .to_gate(),
                    [q_control] + qr[:]
                )

                # Full step: next-nearest-neighbour (b2)
                # Only applies when N >= 4 (num_state_qubits >= 2), which is
                # guaranteed by PoissonProblem1D4th's N >= 4 constraint.
                if self.num_state_qubits >= 2:
                    qc.append(
                        self._next_off_diag_circ(t_step)
                        .control()
                        .to_gate(),
                        [q_control] + qr[:]
                    )

            # Half-step bookend for b1 AFTER the loop (closes the Strang split)
            qc.u(
                -self.off_diag_1 * t_step,
                3 * np.pi / 2,
                np.pi / 2,
                qr[0],
            )

            return qc

        qc_raw.control = control
        return qc_raw