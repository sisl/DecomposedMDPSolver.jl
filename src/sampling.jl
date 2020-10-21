# Function to create matrix from array of arrays. This version is faster than vcat
function fast_mat(S)
    X = Array{Float32, 2}(undef, length(S[1]), length(S))
    for i=1:length(S)
        X[:,i] = S[i]
    end
    X
end

# This function samples episodes from the policy using importance sampling to weight them
function sample_episodes(mdp, policy::Policy, Nsamps::Int64; max_steps::Int64 = 1000,
                         a_and_p = action_and_probability, ϵ = 0)
    S, G, avg_ep_return = [], Float32[], 0.
    Neps = 0
    while length(G) < Nsamps
        Neps += 1
        s, steps = rand(initialstate(mdp)), 0
        Si, Ri, ρi = [], [], []
        push!(Si, convert_s(Array{Float64,1}, s, mdp))
        try
            push!(Ri, reward(mdp, s))
        catch
            push!(Ri, 0.)
        end
        while !isterminal(mdp, s)
            if  rand() < ϵ
                a, prob = rand(actions(mdp)), 1. / length(actions(mdp))
            else
                a, prob = a_and_p(policy, s)
            end
            push!(ρi, policy.action_probability(mdp, s, a) / prob)
            s, r = gen(mdp, s, a)
            push!(Si, convert_s(Array{Float64,1}, s, mdp))
            push!(Ri, r)
            steps += 1
            steps >= max_steps && break
        end
        push!(ρi, 1.)
        steps >= max_steps && println("Episode timeout at ", max_steps, " steps")
        weighted_returns = reverse(cumsum(reverse(Ri))) .* reverse(cumprod(reverse(ρi)))
        push!(S, Si...)
        push!(G, weighted_returns...)
        avg_ep_return += sum(Ri)
    end
    S, G, avg_ep_return = fast_mat(S), G', avg_ep_return/Neps
    @assert size(S,2) == size(G,2)
    @assert size(G,1) == 1
    S, G, avg_ep_return
end

