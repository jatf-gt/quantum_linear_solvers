# This code is an adaptation of the TridiagonalToeplitz class from the
# quantum_linear_solvers library (Vázquez et al., Apache 2.0 licence).
# Original copyright: (C) Copyright IBM 2020, 2021.
#
# Modifications for the pentadiagonal (fourth-order) case:
#   - Added next-nearest-neighbour off-diagonal (b2, ±2 diagonals)
#   - Generalised _off_diag_circ to accept a stride parameter
#   - Updated eigs_bounds for the five-point stencil eigenvalue formula
#   - Updated power() to include both off-diagonal Trotter terms
#   - Corrected ancilla register sizing to match the original exactly
"""Hamiltonian simulation of pentadiagonal Toeplitz symmetric matrices."""

from typing import Tuple

import numpy as np
from scipy.sparse import diags
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import UGate, MCMTVChain

from .linear_system_matrix import LinearSystemMatrix


class PentadiagonalToeplitz(LinearSystemMatrix):
    r"""Hamiltonian simulation for pentadiagonal Toeplitz symmetric matrices.

    Given main diagonal entry :math:`a`, nearest-neighbour off-diagonal
    :math:`b_1`, and next-nearest-neighbour off-diagonal :math:`b_2`, the
    matrix is

    .. math::

        A = a I + b_1 (S + S^\dagger) + b_2 (S^2 + S^{\dagger 2})

    where :math:`S` is the shift operator.  The Hamiltonian simulation uses
    the first-order Lie–Trotter product:

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

    # ── Properties ────────────────────────────────────────────────────────────

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

    # ── Matrix and eigenvalue utilities ───────────────────────────────────────

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
        """Return lower and upper bounds on the absolute eigenvalues.

        Eigenvalues of the N×N Dirichlet pentadiagonal Toeplitz matrix:

            λ_k = a + 2·b1·cos(k·π/(N+1)) + 2·b2·cos(2k·π/(N+1))

        for k = 1, ..., N.  All N values are evaluated exactly.
        """
        n_b = 2 ** self.num_state_qubits
        k = np.arange(1, n_b + 1)
        theta = k * np.pi / (n_b + 1)
        eigs = (
            self.main_diag
            + 2.0 * self.off_diag_1 * np.cos(theta)
            + 2.0 * self.off_diag_2 * np.cos(2.0 * theta)
        )
        abs_eigs = np.abs(eigs)
        return float(abs_eigs.min()), float(abs_eigs.max())

    def condition_bounds(self) -> Tuple[float, float]:
        kappa = np.linalg.cond(self.matrix)
        return kappa, kappa

    # ── Configuration and register management ────────────────────────────────

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
        """Reset quantum registers — identical to TridiagonalToeplitz."""
        qr_state = QuantumRegister(num_state_qubits, "state")
        self.qregs = [qr_state]
        self._ancillas = []
        self._qubits = qr_state[:]
        if num_state_qubits > 1:
            qr_ancilla = AncillaRegister(max(1, num_state_qubits - 1))
            self.add_register(qr_ancilla)

    def _build(self) -> None:
        if self._is_built:
            return
        super()._build()
        self.compose(self.power(1), inplace=True)

    # ── Circuit building blocks ───────────────────────────────────────────────

    def _main_diag_circ(self, theta: float = 1) -> QuantumCircuit:
        """Circuit for e^{i·a·I·theta} — identical to TridiagonalToeplitz."""
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
        self,
        theta: float,
        off_diag_val: float,
        stride: int,
        name: str,
    ) -> QuantumCircuit:
        """Circuit for e^{i·b·(S^stride + S†^stride)·theta}.

        For stride=1 this reproduces TridiagonalToeplitz._off_diag_circ
        exactly, including the unconditional boundary rotation on qr[0].

        For stride=2 the unconditional boundary rotation is OMITTED because
        the j=0 term of the S^2 ladder involves qr[0] AND qr[2] together —
        it is handled by the i=0 iteration of the main loop (which applies
        a CX from qr[0] to qr[2] and then a controlled-U on qr[0]).

        Ancilla sizing matches the original exactly:
          - Uncontrolled circuit: max(1, n-2) ancillas
          - Controlled circuit:   max(1, n-1) ancillas  (one extra for control)
        """
        theta_val = theta * off_diag_val
        n = self.num_state_qubits

        qr = QuantumRegister(n)
        if n > 1:
            # Correct ancilla count: n-2, matching TridiagonalToeplitz exactly
            qr_ancilla = AncillaRegister(max(1, n - 2))
            qc = QuantumCircuit(qr, qr_ancilla, name=name)
        else:
            qc = QuantumCircuit(qr, name=name)
            qr_ancilla = None

        # ── Unconditional boundary rotation (stride=1 only) ───────────────────
        # For stride=1 this is the j=0 term: |0><1| + |1><0| on qr[0] alone.
        # For stride=2 the j=0 term is |0><2| + |2><0|, which requires two
        # qubits and is handled by the i=0 loop iteration below.
        if stride == 1:
            qc.u(-2 * theta_val, 3 * np.pi / 2, np.pi / 2, qr[0])

        # ── Controlled-rotation ladder ────────────────────────────────────────
        # For stride s, the hopping term |j><j+s| + |j+s><j| for j=0..n-s-1
        # is implemented by:
        #   1. CX from qr[i] to qr[i+stride]  (set up parity)
        #   2. X on qr[i]; CX chain from qr[i] back to qr[0..i-1]
        #      (accumulate parity of qubits 0..i-1, controlled-by-0 on qr[i])
        #   3. Multi-controlled U rotation on qr[i]
        #   4. Uncompute steps 1-2
        for i in range(0, n - stride):
            q_controls = []

            qc.cx(qr[i], qr[i + stride])
            q_controls.append(qr[i + stride])

            qc.x(qr[i])
            for j in range(i, 0, -1):
                qc.cx(qr[i], qr[j - 1])
                q_controls.append(qr[j - 1])
            qc.x(qr[i])

            if len(q_controls) > 1:
                ugate = UGate(-2 * theta_val, 3 * np.pi / 2, np.pi / 2)
                qc.append(
                    MCMTVChain(ugate, len(q_controls), 1),
                    q_controls[:] + [qr[i]]
                    + (qr_ancilla[: len(q_controls) - 1] if qr_ancilla else []),
                )
            else:
                qc.cu(
                    -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                    q_controls[0], qr[i],
                )

            # Uncompute
            qc.x(qr[i])
            for j in range(0, i):
                qc.cx(qr[i], qr[j])
            qc.x(qr[i])
            qc.cx(qr[i], qr[i + stride])

        # ── Controlled version (used inside QPE) ──────────────────────────────
        def control(num_ctrl_qubits=1, label=None, ctrl_state=None):
            qr_state = QuantumRegister(n + 1)
            if n > 1:
                # Controlled circuit gets n-1 ancillas (one more than uncontrolled)
                qr_anc = AncillaRegister(max(1, n - 1))
                qc_ctrl = QuantumCircuit(qr_state, qr_anc, name=name)
            else:
                qc_ctrl = QuantumCircuit(qr_state, name=name)
                qr_anc = None

            q_ctrl = qr_state[0]
            qr_d = qr_state[1:]

            # Unconditional rotation on qr_d[0], controlled by q_ctrl
            # (stride=1 only — same logic as uncontrolled version)
            if stride == 1:
                qc_ctrl.cu(
                    -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                    q_ctrl, qr_d[0],
                )

            for i in range(0, n - stride):
                q_controls = [q_ctrl]

                qc_ctrl.cx(qr_d[i], qr_d[i + stride])
                q_controls.append(qr_d[i + stride])

                qc_ctrl.x(qr_d[i])
                for j in range(i, 0, -1):
                    qc_ctrl.cx(qr_d[i], qr_d[j - 1])
                    q_controls.append(qr_d[j - 1])
                qc_ctrl.x(qr_d[i])

                if len(q_controls) > 1:
                    ugate = UGate(-2 * theta_val, 3 * np.pi / 2, np.pi / 2)
                    qc_ctrl.append(
                        MCMTVChain(ugate, len(q_controls), 1).to_gate(),
                        q_controls[:] + [qr_d[i]]
                        + (qr_anc[: len(q_controls) - 1] if qr_anc else []),
                    )
                else:
                    qc_ctrl.cu(
                        -2 * theta_val, 3 * np.pi / 2, np.pi / 2, 0,
                        q_controls[0], qr_d[i],
                    )

                qc_ctrl.x(qr_d[i])
                for j in range(0, i):
                    qc_ctrl.cx(qr_d[i], qr_d[j])
                qc_ctrl.x(qr_d[i])
                qc_ctrl.cx(qr_d[i], qr_d[i + stride])

            return qc_ctrl

        qc.control = control
        return qc

    def _near_off_diag_circ(self, theta: float) -> QuantumCircuit:
        """e^{i·b1·(S+S†)·theta} — nearest-neighbour, stride=1."""
        return self._off_diag_circ(
            theta, self.off_diag_1, stride=1, name="near_off"
        )

    def _next_off_diag_circ(self, theta: float) -> QuantumCircuit:
        """e^{i·b2·(S²+S†²)·theta} — next-nearest-neighbour, stride=2."""
        return self._off_diag_circ(
            theta, self.off_diag_2, stride=2, name="next_off"
        )

    # ── Inverse ───────────────────────────────────────────────────────────────

    def inverse(self) -> "PentadiagonalToeplitz":
        return PentadiagonalToeplitz(
            self.num_state_qubits,
            self.main_diag,
            self.off_diag_1,
            self.off_diag_2,
            evolution_time=-1.0 * self.evolution_time,
        )

    # ── Power (QPE interface) ─────────────────────────────────────────────────

    def power(self, power: int, matrix_power: bool = False) -> QuantumCircuit:
        """Build the controlled-power circuit used inside QPE.

        Trotter product for the pentadiagonal case:

            e^{iA·t·p} ≈ e^{ia·t·p}
                         · [e^{ib1(S+S†)·t·p/m}
                            · e^{ib2(S²+S†²)·t·p/m}]^m

        The main diagonal commutes with everything and is applied once.
        The two off-diagonal terms are interleaved m times, with a
        Strang-splitting half-step bookend on the b1 (nearest-neighbour)
        term only — exactly as TridiagonalToeplitz does for its single
        off-diagonal term.  The b2 term has no bookend because
        _next_off_diag_circ does not apply an unconditional boundary
        rotation on qr[0].
        """
        qc_raw = QuantumCircuit(self.num_state_qubits)

        def control(num_ctrl_qubits=1, label=None, ctrl_state=None):
            qr_state = QuantumRegister(self.num_state_qubits + 1, "state")
            if self.num_state_qubits > 1:
                qr_ancilla = AncillaRegister(
                    max(1, self.num_state_qubits - 1)
                )
                qc = QuantumCircuit(qr_state, qr_ancilla, name="exp(iHk)")
            else:
                qc = QuantumCircuit(qr_state, name="exp(iHk)")
                qr_ancilla = None

            q_control = qr_state[0]
            qr = qr_state[1:]

            # ── Main diagonal: one application at full power·t ────────────────
            qc.append(
                self._main_diag_circ(self.evolution_time * power)
                .control()
                .to_gate(),
                [q_control] + qr[:],
            )

            # ── Trotter steps ─────────────────────────────────────────────────
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
                if qr_ancilla:
                    qc.append(
                        self._near_off_diag_circ(t_step)
                        .control()
                        .to_gate(),
                        [q_control] + qr[:] + qr_ancilla[:],
                    )
                else:
                    qc.append(
                        self._near_off_diag_circ(t_step)
                        .control()
                        .to_gate(),
                        [q_control] + qr[:],
                    )

                # Full step: next-nearest-neighbour (b2)
                # Only applies when N >= 4 (num_state_qubits >= 2), which is
                # guaranteed by PoissonProblem1D4th's N >= 4 constraint.
                if self.num_state_qubits >= 2:
                    if qr_ancilla:
                        qc.append(
                            self._next_off_diag_circ(t_step)
                            .control()
                            .to_gate(),
                            [q_control] + qr[:] + qr_ancilla[:],
                        )
                    else:
                        qc.append(
                            self._next_off_diag_circ(t_step)
                            .control()
                            .to_gate(),
                            [q_control] + qr[:],
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