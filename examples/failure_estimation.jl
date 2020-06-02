using DecomposedMDPSolver
using POMDPs, POMDPModels, POMDPSimulators, POMDPModelTools
using LocalFunctionApproximation, GridInterpolations, LocalApproximationPolicyEvaluation
using Flux
using Random

# Step 1 - Setup the gridworld problem
action_probability(mdp::SimpleGridWorld, s, a) = 0.25
POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, mdp::SimpleGridWorld) = SVector{4, Float64}(1., s[1], s[2], s[1]*s[2])
POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(mdp, s, a )), r=reward(mdp, s, a))
POMDPs.initialstate(mdp::SimpleGridWorld) = rand(initialstate_distribution(mdp))
g_size = (9,9)
g = SimpleGridWorld(size = g_size, rewards = Dict(GWPos(g_size...) => 1, GWPos(1,1) => 0), tprob = 1., discount=1)


# Step 2 - Solve the problem exactly using local approximation
grid = RectangleGrid([1:g_size[1] ...], [1:g_size[2]...])
interp = LocalGIFunctionApproximator(grid)
solver = LocalPolicyEvalSolver(interp, action_probability, is_mdp_generative = true, n_generative_samples = 1,  verbose = true, max_iterations = 2000, belres = 1e-6)
dp_policy = solve(solver, g)
render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(value(dp_policy, s) - 0.5))

# Step 3 - Solve the problem using a MC policy eval with function approximation
V_network = Chain((x) -> x .- 5. ./ 5., Dense(2,32, relu), Dense(32, 1, sigmoid))
render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(V_network(convert_s(Array{Float64, 1}, s, g))[1] - 0.5))

mc_policy_eval!(g, V_network, action_probability, Neps=100, iterations = 10)

render(g, (s=GWPos(1,1), r=1), color = (s) -> 20. *(V_network(convert_s(Array{Float64, 1}, s, g))[1] - 0.5))

