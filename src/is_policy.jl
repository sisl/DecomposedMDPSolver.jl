# uses the A2T style network to create a stochastic policy
struct ISPolicy <: Policy
    network
    mdp
    action_probability::Function
    rng
end

# Get the value of being in state s
function POMDPs.value(p::ISPolicy, s)
    v = p.network(convert_s(Array{Float32,1}, s, p.mdp))
    @assert length(v) == 1
    v[1]
end

# Get the value of being in state s and taking action a
function POMDPs.value(p::ISPolicy, s, a)
    mdp = p.mdp
    sp, r = gen(mdp, s, a, p.rng)
    r + discount(mdp)*(!isterminal(mdp, sp))*value(p, sp)
end

function Distributions.logpdf(policy::ISPolicy, s, a)
    mdp = policy.mdp
    us = [value(policy, s, a)*policy.action_probability(mdp, s, a) for a in actions(policy.mdp, s)]
    if sum(us) == 0
        us = [policy.action_probability(mdp, s, a) for a in actions(policy.mdp, s)]
    end
    p = value(policy, s, a)*policy.action_probability(mdp, s, a) / sum(us)
    return (p == 0) ? log(policy.action_probability(mdp, s, a)) : log(p)
end


Distributions.logpdf(policy::ISPolicy, h::SimHistory) = sum([logpdf(policy, s, a) for (s,a) in eachstep(h, (:s, :a))])


# Choose an action based on the probability of failure. Return action and prob
function action_and_probability(p::ISPolicy, s)
    mdp = p.mdp
    vs = [p.action_probability(mdp, s, a)*value(p, s, a) for a in actions(mdp, s)]
    @assert all(vs .>= 0)
    all(vs .== 0) && (vs = [p.action_probability(mdp, s, a) for a in actions(mdp, s)])
    vs ./= sum(vs)
    ai = rand(p.rng, Categorical(vs))
    actions(mdp)[ai], vs[ai]
end

# Choose an action based on the probability of failure.
POMDPs.action(p, s) = action_and_probability(p, s)[1]

