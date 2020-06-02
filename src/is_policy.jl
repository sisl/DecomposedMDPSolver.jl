# uses the A2T style network to create a stochastic policy
struct ISPolicy <: Policy
    network
    mdp
    action_probability::Function
    rng
end

# Get the value of being in state s
function POMDPs.value(p::ISPolicy, s)
    v = p.network(convert_s(Array{Float64,1}, s, p.mdp))
    @assert length(v) == 1
    v[1]
end

# Get the value of being in state s and taking action a
function POMDPs.value(p::ISPolicy, s, a)
    mdp = p.mdp
    sp, r = gen(mdp, s, a, p.rng)
    r + discount(mdp)*(!isterminal(mdp, sp))*value(p, sp)
end

# Choose an action based on the probability of failure. Return action and prob
function action_and_probability(p::ISPolicy, s)
    mdp = p.mdp
    vs = [p.action_probability(mdp, s, a)*value(p, s, a) for a in actions(mdp, s)]
    @assert all(vs .>= 0)
    all(vs .== 0) && (vs = ones(length(vs)))
    vs ./= sum(vs)
    ai = rand(p.rng, Categorical(vs))
    actions(mdp)[ai], vs[ai]
end

# Choose an action based on the probability of failure.
POMDPs.action(p, s) = action_and_probability(p, s)[1]

