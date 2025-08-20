# Simplex Learning Protocol Implementation

# Protocol struct
struct SimplexProtocol <: DerivativeFreeProtocol
    maxiters::Int
    print_frequency::Int
    # NLopt hyperparameters
    stopval::Union{Float64, Nothing}
    xtol_rel::Float64
    xtol_abs::Float64
    constrtol_abs::Float64
    initial_step::Union{Vector{Float64}, Nothing}
end

SimplexProtocol(; maxiters=200000, print_frequency=20, stopval=nothing, 
                xtol_rel=1e-6, xtol_abs=1e-6, constrtol_abs=1e-8, initial_step=nothing) =
    SimplexProtocol(maxiters, print_frequency, stopval, xtol_rel, xtol_abs, constrtol_abs, initial_step)

# Simplex implementation
function learn(protocol::SimplexProtocol, learning_problem, p_repr_ig; 
               logging::Bool=false, callback_config::CallbackConfig=CallbackConfig())
    
    loss_history = Float32[]
    parameter_history = logging ? [] : nothing
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if callback_config.verbose && current_iter % protocol.print_frequency == 0
            qdrms = sqrt(lossval / callback_config.constants.qdrms_divisor)
            println("In simplex, current loss after $current_iter iterations: $lossval, qdrms=$qdrms at $(now())")
        end
        
        if logging
            push!(parameter_history, deepcopy(p))
        end
        
        return false
    end

    # @show learning_problem.p_repr_lb
    # @show learning_problem.p_repr_ub

    function classical_loss(x, p)
        p_repr = reconstruct_learning_params_from_array(x, p_repr_ig, learning_problem.model)
        loss = get_loss(p_repr; learning_problem=learning_problem)
        
        # Debug: Check for stale flexi parameters
        # if length(x) > 0 && haskey(p_repr_ig, :flex1_params) && !isempty(p_repr_ig.flex1_params)
        #     println("WEDNESDAY DEBUGGING: classical_loss evaluation")
        #     println("  Classical param checksum: $(sum(abs.(x)))")
        #     println("  Flexi1 param checksum: $(sum(abs.(p_repr_ig.flex1_params)))")
        #     println("  Flexi2 param checksum: $(sum(abs.(p_repr_ig.flex2_params)))")
        #     println("  Loss: $loss")
        # end
        
        return loss
    end



    # Convert ComponentArray to Array for NLopt compatibility (like BBO)
    # CRITICAL: Must maintain consistent ordering between params and bounds!
    
    # Get the keys in the same order for both params and bounds
    param_keys = collect(keys(p_repr_ig.p_classical))
    # println("Parameter keys order: $param_keys")
    
    # Convert params using consistent key ordering
    classical_params_array = [p_repr_ig.p_classical[key] for key in param_keys]

    # @show classical_params_array
    # @show p_repr_ig.p_classical

    # Extract only classical bounds to match classical parameters array dimensions
    # println("=== Bounds Extraction Debug ===")
    # println("learning_problem.p_repr_lb type: $(typeof(learning_problem.p_repr_lb))")
    # println("learning_problem.p_repr_ub type: $(typeof(learning_problem.p_repr_ub))")
    # @show learning_problem.p_repr_lb
    # @show learning_problem.p_repr_ub
    
    # For flexi models, bounds structure is: ComponentArray(p_classical=..., flex1_params=..., flex2_params=...)
    # For classical models, bounds structure is: ComponentArray(p_classical=...)
    if haskey(learning_problem.p_repr_lb, :p_classical)
        println("Extracting classical bounds from p_classical field...(Jun doesn't think this should happen)")
        # Use same key ordering for bounds as we used for params
        classical_lb = [learning_problem.p_repr_lb.p_classical[key] for key in param_keys]
        classical_ub = [learning_problem.p_repr_ub.p_classical[key] for key in param_keys]
    else
        # println("Using direct bounds extraction (classical model)...")
        # Use same key ordering for bounds as we used for params
        classical_lb = [learning_problem.p_repr_lb[key] for key in param_keys]
        classical_ub = [learning_problem.p_repr_ub[key] for key in param_keys]
    end
    # println("=== End Bounds Extraction Debug ===")

    # @show classical_params_array
    # @show classical_lb
    # @show classical_ub

    # @show classical_loss(classical_params_array, nothing)
    # @show classical_loss(classical_lb, nothing)
    # @show classical_loss(classical_ub, nothing)

    # Additional NLopt diagnostics
    # println("=== NLopt INVALID_ARGS Diagnostics ===")
    # println("Parameter array length: $(length(classical_params_array))")
    # println("Lower bounds length: $(length(classical_lb))")
    # println("Upper bounds length: $(length(classical_ub))")
    
    # Check for dimension mismatches
    if length(classical_params_array) != length(classical_lb) || length(classical_params_array) != length(classical_ub)
        error("DIMENSION MISMATCH: params=$(length(classical_params_array)), lb=$(length(classical_lb)), ub=$(length(classical_ub))")
    end
    
    # Check for invalid bounds relationships
    for i in 1:length(classical_params_array)
        if classical_lb[i] >= classical_ub[i]
            error("INVALID BOUNDS at index $i: lb=$(classical_lb[i]) >= ub=$(classical_ub[i])")
        end
        if classical_params_array[i] < classical_lb[i] || classical_params_array[i] > classical_ub[i]
            println("WARNING: Initial param $i ($(classical_params_array[i])) outside bounds [$(classical_lb[i]), $(classical_ub[i])]")
        end
    end
    
    # Check for NaN/Inf values
    if any(isnan, classical_params_array) || any(isinf, classical_params_array)
        error("INVALID VALUES in params: $(classical_params_array)")
    end
    if any(isnan, classical_lb) || any(isinf, classical_lb)
        error("INVALID VALUES in lower bounds: $(classical_lb)")
    end
    if any(isnan, classical_ub) || any(isinf, classical_ub)
        error("INVALID VALUES in upper bounds: $(classical_ub)")
    end
    
    # Test direct NLopt call
    # println("Testing direct NLopt.LN_SBPLX creation...")
    # try
    #     # Direct NLopt test with same dimensions
    #     opt = NLopt.Opt(:LN_SBPLX, length(classical_params_array))
    #     NLopt.lower_bounds!(opt, classical_lb)
    #     NLopt.upper_bounds!(opt, classical_ub)
    #     println("✓ Direct NLopt.LN_SBPLX creation successful")
    # catch e
    #     println("✗ Direct NLopt.LN_SBPLX creation failed: $e")
    # end
    
    # println("=== End NLopt Diagnostics ===")

    # error("Stop here for debugging")

    # Create optimization problem with bounds for NLopt.LN_SBPLX()
    prob = Optimization.OptimizationProblem(classical_loss, classical_params_array; 
                                           lb=classical_lb, ub=classical_ub)

    # @show prob

    # Create NLopt optimizer with custom hyperparameters
    opt = NLopt.LN_SBPLX()
    
    # Apply hyperparameters if specified
    if protocol.stopval !== nothing
        # Note: stopval would be applied via solve() options in Optimization.jl
        # For now, we'll pass it through the solve call
    end
    
    # Note: NLopt tolerance settings are handled by Optimization.jl wrapper
    # These will be applied via solve() options
    solve_options = Dict{Symbol, Any}()
    if protocol.stopval !== nothing
        solve_options[:abstol] = protocol.stopval
    end
    solve_options[:reltol] = protocol.xtol_rel
    
    sol = solve(prob, opt; callback=callback, maxiters=protocol.maxiters, solve_options...)
    # sol = solve(prob, NLopt.LN_NELDERMEAD(); callback=callback, maxiters=protocol.maxiters)

    # @show sol.minimizer

    # Convert back from Array to ComponentArray (like BBO)
    final_params = reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)

    # @show final_params

    convergence_status = sol.retcode == :success ? :converged : :max_iterations
    
    return create_learning_result("simplex", final_params, loss_history, 
                                 parameter_history, learning_problem; 
                                 convergence_status=convergence_status)
end
