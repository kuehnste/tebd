# tebd

Simple time evolution code for Matrix Product States (MPS) with open boundary conditions written in the [Julia Programming Language](https://julialang.org/).

## Overview

This repository accompanies the hands-on tutorials for the lectures on Tensor Networks at the [CERN-MPQ-UIBK School on Quantum Simulation of Fundamental Physics](https://indico.cern.ch/event/1623729/). 

The project provides a simple, pedagogical implementation of an algorithm for simulating the time evolution of one-dimensional quantum many-body systems MPS. The implementation is intended primarily for educational purposes and follows the exercises from the accompanying tutorial. It includes functionality for both real-time and imaginary-time evolution and assumes open boundary conditions.

## Repository Structure

```
project/
│
├── src/            Source code
├── examples/       Example programs
├── tests/          Unit tests
├── LICENSE
└── README.md
```
The folder `src` contains the codes and different functions that are created by solving the problems on the exercise sheet. The folder `examples` contains a simple example of running the TEBD code in both real and imaginary time.


## Installation

To run the code, you will need to have Julia installed. Julia can be downloaded from the [official website](https://julialang.org/downloads/). 

For instructions on getting started, see the [Julia documentation](https://docs.julialang.org/en/v1/manual/getting-started/).


## Contributing

Contributions are welcome.

Please
- open an issue for bugs,
- discuss major changes before implementing them,
- follow the existing coding style.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

There is a vast amount of literature on MPS and more general tensor network methods available. The following incomplete list of reviews and references therein might provide a useful starting point for learning more about MPS and more general tensor networks.

* F. Verstraete, V. Murg, J. Cirac, [Adv. Phys. 57, 143 (2008)](https://doi.org/10.1080/14789940801912366)
* U. Schollwöck, [Ann. Phys. 326, January 2011 Special Issue, 96 (2011)](https://doi.org/10.1016/j.aop.2010.09.012)
* R. Orús, [Ann. Phys. 349, 117 (2014)](https://doi.org/10.1016/j.aop.2014.06.013)
* J. C. Bridgeman, C. T. Chubb, [J. Phys. A 50, 223001 (2017)](https://doi.org/10.1088/1751-8121/aa6dc3)
* M. C. Banuls, [Lecture notes](https://drive.google.com/file/d/1FOWJwbwyErND7XCjF5fyQ1bGHGrKTaif/view)

## Authors

[Stefan Kühn](https://github.com/kuehnste)
