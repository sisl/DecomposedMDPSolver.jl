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

a2t_model(rand(4))
input = rand(4)
i = 1
s = 10
a2t_model.solutions[s](input[:, i])

## Constant Weights
base = Chain(Dense(4, 32, relu), Dense(32, Na))
attn = Chain(ConstantLayer(Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)
@test size(a2t_model(rand(4))) == (Na, 1)

## Version that was failing -- local approx policy eval

sols = rand(100)
function val_fn(s)
    sols[1] = rand()
    return sols[1]
end
base = Chain(Dense(2, 32, relu), Dense(32, 1, σ))
attn = Chain(Dense(2, 32, relu), Dense(32, 2, exp))
solutions = [val_fn]

model = A2TNetwork(base, attn, solutions)

for l in model
    @test !isnothing(l)
end


S, G = rand(2, 100), rand(1,100)
data = Flux.Data.DataLoader((S, G), batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, opt)


## Testing the state transformation network
base = Chain(Dense(2, 32, relu), Dense(32, 1, σ))
attn = Chain(Dense(2, 32, relu), Dense(32, 3, exp))
strans = [Chain(Dense(2, 32, relu), Dense(32, 2, relu)), Chain(Dense(2, 32, relu), Dense(32, 2, relu))]
solutions = [Chain(Dense(2, 32, relu), Dense(32, 1, σ)), Chain(Dense(2, 32, relu), Dense(32, 1, σ))]

model = A2TSTNetwork(base, attn, strans, solutions)
@test size(strans[1](rand(2))) == (2,)
@test size(strans[1](rand(2, 100))) == (2,100)
@test size(model(rand(2, 100))) == (1,100)

i = rand(2, 100)
v = [s(i) for s in model.solutions]
out = model(i)

S, G = rand(2, 100), rand(1,100)
data = Flux.Data.DataLoader((S, G), batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, opt)

v2 = [s(i) for s in model.solutions]
out2 = model(i)

@test v2 == v
@test all(out2 .!= out)
for l in model
    @test !isnothing(l)
end


## Test the fine-tune A2T network

base = Chain(Dense(4, 32, relu), Dense(32, 9, σ))
attn = Chain(Dense(4, 32, relu), Dense(32, 3), softmax)
solutions = [Chain(Dense(4, 32, relu), Dense(32, 9, σ)), Chain(Dense(4, 32, relu), Dense(32, 9, σ))]
finetune = [Chain(Dense(13, 9)), Chain(Dense(13, 9))]

model = A2TFTNetwork(base, attn, solutions, finetune)

model(rand(4))
model(rand(4, 100))

S, G = rand(4, 100), rand(9,100)

v = [s(S) for s in model.solutions]
out = model(S)
val = Flux.mse(model(S), G)

data = Flux.Data.DataLoader((S, G), batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, opt)

v2 = [s(S) for s in model.solutions]
out2 = model(S)
val2 = Flux.mse(model(S), G)

@test val2 < val
@test all(out[:] != out2[:])
@test all( v .== v2)

for l in model
    @test !isnothing(l)
end

@test deepcopy(model) isa A2TFTNetwork


## Test the fine-tune network
base = Chain(Dense(2, 2, sigmoid), Dense(2, 2, sigmoid), Dense(2,2))
mynet = FTNetwork(base, [1,2])

params_to_change = deepcopy(params(mynet.net))[1:4]
params_to_stay_same = deepcopy(params(mynet.net))[5:6]

@test length(Flux.trainable(mynet)) == 2

S = rand(2,100)
G = rand(2, 100)


data = Flux.Data.DataLoader((S, G), batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(mynet(x), y), Flux.params(mynet), data, opt)

@test all(params_to_change .!= params(mynet.net)[1:4])
@test all(params_to_stay_same .== params(mynet.net)[5:6])


## Full test with A2T network + finetuning
base = Chain(Dense(4, 32, sigmoid), Dense(32, 9, σ))
attn = Chain(Dense(4, 32, sigmoid), Dense(32, 3), softmax)
solutions = [Chain(Dense(4, 32, sigmoid), Dense(32, 9, σ)), Chain(Dense(4, 32, sigmoid), Dense(32, 9, σ))]
mysols = [FTNetwork(net, [2]) for net in solutions]
model = A2TNetwork(base, attn, mysols, false)

base_params = deepcopy(params(model.base))
@test all(base_params .== params(model.base))
attn_params = deepcopy(params(model.attn))
@test all(attn_params .== params(model.attn))
params_to_change1 = deepcopy(params(model.solutions[1].net))[3:4]
@test all(params_to_change1 .== params(model.solutions[1].net)[3:4])
params_to_change2 = deepcopy(params(model.solutions[2].net))[3:4]
@test all(params_to_change2 .== params(model.solutions[2].net)[3:4])
params_to_stay_same1 = deepcopy(params(model.solutions[1].net))[1:2]
@test all(params_to_stay_same1 .== params(model.solutions[1].net)[1:2])
params_to_stay_same2 = deepcopy(params(model.solutions[2].net))[1:2]
@test all(params_to_stay_same2 .== params(model.solutions[2].net)[1:2])

@test length(Flux.trainable(model)) == 4
S = rand(4, 100)
G = rand(9, 100)

@test length(Flux.params(model)) == 12

data = Flux.Data.DataLoader((S, G), batchsize=32, shuffle = true)
opt = ADAM()
Flux.train!((x, y) -> Flux.mse(model(x), y), Flux.params(model), data, opt)

@test all(base_params .!= params(model.base))
@test all(attn_params .!= params(model.attn))
@test all(params_to_change1 .!= params(model.solutions[1].net)[3:4])
@test all(params_to_change2 .!= params(model.solutions[2].net)[3:4])
@test all(params_to_stay_same1 .== params(model.solutions[1].net)[1:2])
@test all(params_to_stay_same2 .== params(model.solutions[2].net)[1:2])

