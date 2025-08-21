# CMA-ES Learning Protocol Implementation

# Protocol struct
struct CMAESProtocol <: DerivativeFreeProtocol
    maxiters::Int
    mu::Int
    lambda::Int
    upper_bound_multiplier::Float64
    # CMAES hyperparameters
    c_1::Float64
    c_c::Float64
    c_mu::Float64
    c_sigma::Float64
    c_m::Float64
    sigma0::Float64
    weights::Union{Vector{Float64}, Nothing}
end

CMAESProtocol(; maxiters=10, mu=40, lambda=100, upper_bound_multiplier=10.0,
              c_1=0.0, c_c=0.0, c_mu=0.0, c_sigma=0.0, c_m=1.0, sigma0=-1, weights=nothing) =
    CMAESProtocol(maxiters, mu, lambda, upper_bound_multiplier, c_1, c_c, c_mu, c_sigma, c_m, sigma0, weights)

# Compute adaptive sigma0 based on initial guess characteristics
function compute_adaptive_sigma0(ig, protocol)
    # Method 1: Scale based on parameter magnitudes (robust to parameter scales)
    # Use a fraction of the typical parameter magnitude
    nonzero_params = ig[abs.(ig) .> 1e-12]  # Exclude near-zero parameters
    if length(nonzero_params) > 0
        typical_magnitude = mean(abs.(nonzero_params))
        magnitude_based = typical_magnitude * 0.1  # 10% of typical parameter size
    else
        magnitude_based = 0.1  # Fallback if all parameters are near zero
    end
    
    # Method 2: Scale based on current bound setup
    flexi_bound = maximum(abs.(ig)) * protocol.upper_bound_multiplier
    bound_based = flexi_bound * 0.02  # 2% of the dynamic bound
    
    # Method 3: Scale based on parameter variance
    param_std = std(ig)
    variance_based = param_std * 0.5  # Half the parameter standard deviation
    
    # Combine methods with weights favoring magnitude-based approach
    combined_sigma0 = 0.6 * magnitude_based + 0.3 * bound_based + 0.1 * variance_based
    
    # Apply reasonable bounds
    min_sigma0 = 1e-4
    max_sigma0 = 1.0
    
    return clamp(combined_sigma0, min_sigma0, max_sigma0)
end

# CMA-ES implementation
function learn(protocol::CMAESProtocol, learning_problem, p_repr_ig; 
               logging::Bool=false, callback_config::CallbackConfig=CallbackConfig())
    
    # Check if model has flex2_params once for performance
    has_flex2 = haskey(p_repr_ig, :flex2_params)
    
    if has_flex2
        flex_all = ComponentArray(
            flex1_params=deepcopy(p_repr_ig.flex1_params), 
            flex2_params=deepcopy(p_repr_ig.flex2_params)
        )
    else
        flex_all = ComponentArray(
            flex1_params=deepcopy(p_repr_ig.flex1_params)
        )
    end

    # Define loss function that handles both cases
    function flexi_loss(flexi_x_array, p)
        if has_flex2
            full_params = ComponentArray(
                p_classical=p_repr_ig.p_classical,
                flex1_params=flexi_x_array[1:length(p_repr_ig.flex1_params)],
                flex2_params=flexi_x_array[(length(p_repr_ig.flex1_params) + 1):end],
            )
        else
            full_params = ComponentArray(
                p_classical=p_repr_ig.p_classical,
                flex1_params=flexi_x_array,
            )
        end
        return get_loss(full_params; learning_problem=learning_problem)
    end

    # Track best solution during optimization
    best_loss = Inf
    best_flexi_params = nothing
    
    # Evaluate initial guess
    ig = deepcopy(collect(values(copy((flex_all)))))
    initial_loss = flexi_loss(ig, nothing)
    # best_loss = initial_loss
    # best_flexi_params = deepcopy(ig)
    
    # Create custom callback that tracks the best solution
    loss_history = Float32[]
    parameter_history = logging ? [] : nothing
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        # Track best solution encountered - use fresh loss evaluation instead of lossval
        if !hasfield(typeof(p), :u)
            error("WEDNESDAY DEBUGGING: CMA-ES callback parameter missing :u field, type: $(typeof(p))")
        end
        
        # # Evaluate fresh loss at current parameters
        fresh_loss = flexi_loss(p.u, nothing)
        # println("WEDNESDAY DEBUGGING: CMA-ES callback iter $current_iter - lossval=$lossval, fresh_loss=$fresh_loss, param_sum=$(sum(abs.(p.u)))")
        
        if fresh_loss < best_loss
            best_loss = fresh_loss
            best_flexi_params = deepcopy(p.u)
            println("New best solution found at iter $current_iter with loss $fresh_loss")
        end
        
        if callback_config.verbose && current_iter % callback_config.print_frequency == 0
            qdrms = sqrt(lossval / callback_config.constants.qdrms_divisor)
            println("In cmaes-on-flexi, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        
        if logging
            push!(parameter_history, deepcopy(p))
        end
        
        return false
    end

    optf = Optimization.OptimizationFunction(flexi_loss)

    # Set up bounds for flexi parameters
    flexi_bound = maximum(abs.(ig)) .* protocol.upper_bound_multiplier 

    lb = 0.0*fill(flexi_bound, length(ig));#-1.0*fill(flexi_bound, length(ig))
    ub = +1.0*fill(flexi_bound, length(ig))

    prob = Optimization.OptimizationProblem(optf, ig, [1.0, 100.0]; lb=lb, ub=ub)

    # Build CMAES options with hyperparameters
    cmaes_options = Dict{Symbol, Any}()
    cmaes_options[:μ] = protocol.mu
    cmaes_options[:λ] = protocol.lambda
    
    # Add hyperparameters if they differ from defaults
    if protocol.c_1 != 0.0
        cmaes_options[:c_1] = protocol.c_1
    end
    if protocol.c_c != 0.0
        cmaes_options[:c_c] = protocol.c_c
    end
    if protocol.c_mu != 0.0
        cmaes_options[:c_mu] = protocol.c_mu
    end
    if protocol.c_sigma != 0.0
        cmaes_options[:c_sigma] = protocol.c_sigma
    end
    if protocol.c_m != 1.0
        cmaes_options[:c_m] = protocol.c_m
    end
    
    # Handle sigma0: use adaptive computation if sigma0 = -1, otherwise use specified value
    if protocol.sigma0 == -1.0
        adaptive_sigma0 = compute_adaptive_sigma0(ig, protocol)
        cmaes_options[:sigma0] = adaptive_sigma0
        println("Using adaptive sigma0 = $adaptive_sigma0 (computed from initial guess)")
    elseif protocol.sigma0 != 1.0
        cmaes_options[:sigma0] = protocol.sigma0
    end
    
    if protocol.weights !== nothing
        cmaes_options[:weights] = protocol.weights
    end
    
    sol = solve(prob, Evolutionary.CMAES(; cmaes_options...); 
                callback=callback, maxiters=protocol.maxiters)

    # Determine which solution to return: initial guess vs best found vs final solution
    final_loss_from_sol = flexi_loss(collect(sol.minimizer), nothing)
    
    # Choose the best among: initial guess, best during optimization, final solution
    candidates = [
        (initial_loss, ig, "initial guess"),
        (best_loss, best_flexi_params, "best during optimization"), 
        (final_loss_from_sol, collect(sol.minimizer), "final solution")
    ]
    
    # println("CMA-ES candidate selection")
    # for (i, (loss, params, source)) in enumerate(candidates)
    #     println("  Candidate $i ($source): loss=$loss, param_sum=$(sum(abs.(params)))")
    # end
    
    best_candidate_idx = argmin([x[1] for x in candidates])
    chosen_loss, chosen_flexi_params, chosen_source = candidates[best_candidate_idx]
    
    # println("W. Chosen candidate: $chosen_source with loss $chosen_loss")
    
    println("CMA-ES: Chose $chosen_source with loss $chosen_loss")
    println("  Initial guess loss: $initial_loss")
    println("  Best during optimization: $best_loss") 
    println("  Final solution loss: $final_loss_from_sol")

    if has_flex2
        final_params = ComponentArray(
            p_classical=p_repr_ig.p_classical,
            flex1_params=chosen_flexi_params[1:length(p_repr_ig.flex1_params)],
            flex2_params=chosen_flexi_params[(length(p_repr_ig.flex1_params) + 1):end],
        )
    else
        final_params = ComponentArray(
            p_classical=p_repr_ig.p_classical,
            flex1_params=chosen_flexi_params,
        )
    end

    convergence_status = sol.retcode == :success ? :converged : :max_iterations
    
    return create_learning_result("cmaes-on-flexi", final_params, loss_history, 
                                 parameter_history, learning_problem; 
                                 convergence_status=convergence_status)
end