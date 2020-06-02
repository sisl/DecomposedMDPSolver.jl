using DecomposedMDPSolver
using POMDPs, POMDPModels
using Test
using Random
using Flux

mdp = SimpleGridWorld(tprob = 1, discount = 1)
POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(mdp, s, a )), r=reward(mdp, s, a))

Na = 1
Np = 10
solutions = [(x) -> 0.5*ones(Na) for i=1:Np ]

## Weights network
base = Chain(Dense(2, 32, relu), Dense(32, Na))
attn = Chain(Dense(2, 32, relu), Dense(32, Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)

action_prob(mdp, s, a) = 0.25
p = ISPolicy(a2t_model, mdp, action_prob, Random.GLOBAL_RNG)
@test p.network == a2t_model
@test p.mdp isa SimpleGridWorld
@test p.action_probability(mdp, rand(initialstate_distribution(mdp)), :up) == 0.25
@test p.rng == Random.GLOBAL_RNG

@test length(value(p, GWPos(3,3))) == 1

@test value(p, GWPos(3,3), :up) == value(p, GWPos(3,4))

a, prob = action_and_probability(p, GWPos(2,2))
@test a in [:up, :left, :right, :down]
@test prob isa Float64
@test action(p, GWPos(2,2)) in [:up, :left, :right, :down]



