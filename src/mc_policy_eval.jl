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
                         logger = TBLogger("log/", tb_increment),
                         log_freq = 1,
                         exploration_policy = nothing)
    log = !isnothing(logger)
    loss = (x, y) -> Flux.mse(V(x), y)
    best_avgR = 0
    for iter in 1:iterations
        verbose && println("iteration: ", iter)
        # Setup the exploration policy. If none is provided, use the current policy
        isnothing(exploration_policy) && (exploration_policy = ISPolicy(V, mdp, action_probability, rng))

        # Sample episodes. S is the states, G is the per-state return (with IS weights), R is the undiscounted return for each episode
        S, G, avgR = sample_episodes(mdp, exploration_policy, Neps,
                                 max_steps = max_steps_per_ep, a_and_p = a_and_p)

        # If the returns are higher (i.e. more failures found) then save the model
        log && avgR > best_avgR && @save string(logger.logdir, "/best_model.bson") V

        # Load the data into a DataLoader and train
        data = Flux.Data.DataLoader(S, G, batchsize=batchsize, shuffle = shuffle)
        Flux.train!(loss, Flux.params(V), data, opt)

        # If we are logging then record the loss and the avg return
        (!log || iter % log_freq != 0) && continue
        with_logger(logger) do
          @info "Training" loss=loss(S, G) undiscounted_return = avgR
          @save string(logger.logdir, "/last_model.bson") V
        end
    end
end



