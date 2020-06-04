# Function to create matrix from array of arrays. This version is faster than vcat
function fast_mat(S)
    X = Array{Float32, 2}(undef, length(S[1]), length(S))
    for i=1:length(S)
        X[:,i] = S[i]
    end
    X
end

# This function samples episodes from the policy using importance sampling to weight them
function sample_episodes(mdp, policy::Policy, Neps::Int64; max_steps::Int64 = 1000,
                         a_and_p = action_and_probability)
    S, G, ep_return = [], Float32[], []
    for i=1:Neps
        s, steps = initialstate(mdp), 0
        Si, Ri, ρi = [], [], []
        while !isterminal(mdp, s)
            push!(Si, convert_s(Array{Float64,1}, s, mdp))
            a, prob = a_and_p(policy, s)
            push!(ρi, policy.action_probability(mdp, s, a) / prob)
            s, r = gen(mdp, s, a)
            push!(Ri, r)
            steps += 1
            steps >= max_steps && break
        end
        steps >= max_steps && println("Episode timeout at ", max_steps, " steps")
        weighted_returns = reverse(cumsum(reverse(Ri))) .* reverse(cumprod(reverse(ρi)))
        push!(S, Si...)
        push!(G, weighted_returns...)
        push!(ep_return, sum(Ri))
    end
    S, G = fast_mat(S), G'
    @assert size(S,2) == size(G,2)
    @assert size(G,1) == 1
    S, G, ep_return
end

