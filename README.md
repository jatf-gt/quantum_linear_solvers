# Quantum linear solvers (PDE Extension Fork)
This repository is a modified fork of `quantum_linear_solvers` designed to support higher-order finite difference PDE solvers using the HHL algorithm.

**Key additions in this fork:**
- `PentadiagonalToeplitz` class for 4th-order (pentadiagonal) finite difference stencils.

## Installation
```bash
git clone https://github.com/jat125/quantum_linear_solvers.git
cd quantum_linear_solvers
pip install .
```

## Documentation
For original documentation and tutorial, see: https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation
