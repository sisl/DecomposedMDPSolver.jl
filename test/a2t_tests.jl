using DecomposedMDPSolver
using Test
using Flux

Na = 5
Np = 10
solutions = [(x) -> rand(Na) for i=1:Np ]
## Weights network
base = Chain(Dense(4, 32, relu), Dense(32, Na))
attn = Chain(Dense(4, 32, relu), Dense(32, Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)
@test size(a2t_model(rand(4))) == (Na, 1)


## Constant Weights
base = Chain(Dense(4, 32, relu), Dense(32, Na))
attn = Chain(ConstantLayer(Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)
@test size(a2t_model(rand(4))) == (Na, 1)

