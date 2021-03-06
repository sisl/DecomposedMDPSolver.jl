# DecomposedMDPSolver.jl
[![Build Status](https://travis-ci.org/sisl/DecomposedMDPSolver.jl.svg?branch=master)](https://travis-ci.org/sisl/DecomposedMDPSolver.jl) [![Coverage Status](https://coveralls.io/repos/github/sisl/DecomposedMDPSolver.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/DecomposedMDPSolver.jl?branch=master) [![codecov](https://codecov.io/gh/sisl/DecomposedMDPSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/DecomposedMDPSolver.jl)

Tools for solving an MDP using decomposition. The two main contributions are
1. An implementation of the Attend, Adapt and Transfer (A2T) network for Q learning: https://arxiv.org/abs/1510.02879
2. An implementation of Monte-Carlo Policy evaluation

## Usage
1. For A2T, construct an `A2TNetwork` by defining a base network, an attention network, and list of functions that compute estimates to the Q values (either from previous solutions or sub problems)
2. For Monte-Carlo Policy evaluation, see `examples/failure_estimation.jl` to see how to compute the probability of failure using this approach.

Maintained by Anthony Corso (acorso@stanford.edu)
