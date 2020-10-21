function mc_policy_eval!(mdp, V, action_probability::Function;
                         iterations = 100,
                         Nsamps = 320,
                         verbose = true,
                         max_steps_per_ep = 1000,
                         opt = ADAM(1e-3),
                         batchsize = 32,
                         shuffle = true,
                         rng::AbstractRNG = Random.GLOBAL_RNG,
                         a_and_p = action_and_probability,
                         logger = TBLogger("log/", tb_increment),
                         eval_fn = (args...)->0,
                         log_freq = 10,
                         exploration_policy = nothing,
                         ϵstart = 0.,
                         ϵend = 0.,
                         ϵsteps = iterations/2)
    log = !isnothing(logger)
    loss = (x, y) -> Flux.mse(V(x), y)
    best_avgR = 0
    steps_per_iter = Int(Nsamps / batchsize)

    if log
        t = 0
        isnothing(exploration_policy) && (exploration_policy = ISPolicy(V, mdp, action_probability, rng))
        S, G, avgR = sample_episodes(mdp, exploration_policy, Nsamps,
                                 max_steps = max_steps_per_ep, a_and_p = a_and_p, ϵ = ϵstart)
        with_logger(logger) do
            log_value(logger, "eval_reward", eval_fn(exploration_policy), step = t)
            log_value(logger, "avg_reward", avgR, step = t)
            log_value(logger, "loss", loss(S,G), step = t)
            log_value(logger, "eps", ϵstart, step = t)
            end
    end
    for iter in 1:iterations
        t = steps_per_iter * iter
        verbose && println("iteration: ", iter)
        # Setup the exploration policy. If none is provided, use the current policy
        isnothing(exploration_policy) && (exploration_policy = ISPolicy(V, mdp, action_probability, rng))

        # Sample episodes. S is the states, G is the per-state return (with IS weights), R is the undiscounted return for each episode
        ϵ = max(ϵend, ϵstart + (ϵend - ϵstart)*(iter / ϵsteps))
        S, G, avgR = sample_episodes(mdp, exploration_policy, Nsamps,
                                 max_steps = max_steps_per_ep, a_and_p = a_and_p, ϵ = ϵ)

        # If the returns are higher (i.e. more failures found) then save the model
        log && avgR > best_avgR && @save string(logger.logdir, "/best_model.bson") V

        # Load the data into a DataLoader and train
        data = Flux.Data.DataLoader((S, G), batchsize=batchsize, shuffle = shuffle)
        Flux.train!(loss, Flux.params(V), data, opt)

        # If we are logging then record the loss and the avg return
        (!log || iter % log_freq != 0) && continue
        with_logger(logger) do
          log_value(logger, "eval_reward", eval_fn(exploration_policy), step = t)
          log_value(logger, "avg_reward", avgR, step = t)
          log_value(logger, "loss", loss(S,G), step = t)
          log_value(logger, "eps", ϵ, step = t)
          @save string(logger.logdir, "/last_model.bson") V
        end
    end
end



