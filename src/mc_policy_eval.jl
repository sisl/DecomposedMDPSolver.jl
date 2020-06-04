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
                         logdir = "log/",
                         log = true,
                         log_freq = 1,
                         exploration_policy = nothing)
    logger = log ? TBLogger(logdir, tb_increment) : nothing
    loss = (x, y) -> Flux.mse(V(x), y)
    for iter in 1:iterations
        verbose && println("iteration: ", iter)
        isnothing(exploration_policy) && (exploration_policy = ISPolicy(V, mdp, action_probability, rng))
        S, G, R = sample_episodes(mdp, exploration_policy, Neps,
                                 max_steps = max_steps_per_ep, a_and_p = a_and_p)

        data = Flux.Data.DataLoader(S, G, batchsize=batchsize, shuffle = shuffle)
        Flux.train!(loss, Flux.params(V), data, opt)
        (!log || iter % log_freq != 0) && continue
        with_logger(logger) do
          @info "Training" loss=loss(S, G) undiscounted_return = sum(R) / length(R)
        end
    end
end

