using DecomposedMDPSolver
using Flux
using Test

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

## Version that was failing -- local approx policy eval

sols = rand(100)
function val(s)
    sols[1] = rand()
    return sols[1]
end
base = Chain(Dense(2, 32, relu), Dense(32, 1, σ))
attn = Chain(Dense(2, 32, relu), Dense(32, 2, exp))
solutions = [val]

model = A2TNetwork(base, attn, solutions)

S, G = rand(2, 100), rand(1,100)
data = Flux.Data.DataLoader(S, G, batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, opt)



