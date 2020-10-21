using DecomposedMDPSolver
using POMDPs, POMDPModels
using Test
using Random
using Flux

mdp = SimpleGridWorld(tprob = 1, discount = 1, rewards = Dict(GWPos(4,3)=>0., GWPos(4,6)=>0., GWPos(9,3)=>1.0, GWPos(8,8)=>1.0))
POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(mdp, s, a )), r=reward(mdp, s, a))
POMDPs.initialstate(mdp::SimpleGridWorld) = initialstate_distribution(mdp)
Na = 1
Np = 10
solutions = [(x) -> 0.5*ones(Na) for i=1:Np ]

## Weights network
base = Chain(Dense(2, 32, relu), Dense(32, Na, sigmoid))
attn = Chain(Dense(2, 32, relu), Dense(32, Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)

action_prob(mdp, s, a) = 0.25
p = ISPolicy(a2t_model, mdp, action_prob, Random.GLOBAL_RNG)

S, G, R = sample_episodes(mdp, p, 10)
@test all(G .>= 0)
@test size(S, 1) == 2
@test length(G) == size(S, 2)
@test R >= 0

