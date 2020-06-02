function mc_policy_eval!(mdp, V, action_probability::Function;
                         iterations = 100,
                         Neps = 100,
                         verbose = true,
                         max_steps_per_ep = 1000,
                         opt = ADAM(1e-3),
                         batchsize = 32,
                         shuffle = true,
                         rng::AbstractRNG = Random.GLOBAL_RNG,
                         a_and_p = action_and_probability,
                         exploration_policy = nothing)
    for iter in 1:iterations
        verbose && println("iteration: ", iter)
        isnothing(exploration_policy) && (exploration_policy = ISPolicy(V, mdp, action_probability, rng))
        S, G = sample_episodes(mdp, exploration_policy, Neps, max_steps = max_steps_per_ep,
                                a_and_p = a_and_p)

        data = Flux.Data.DataLoader(S, G, batchsize=batchsize, shuffle = shuffle)
        Flux.train!((x, y) -> Flux.mse(V(x), y), Flux.params(V), data, opt)
    end
end

