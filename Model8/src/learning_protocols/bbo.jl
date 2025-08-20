# BBO Learning Protocol Implementation

# Protocol struct
struct BBOProtocol <: DerivativeFreeProtocol
    maxiters::Int
    intermediate_save_function::Union{Function, Nothing}
end

BBOProtocol(; maxiters=200000, intermediate_save=nothing) =
    BBOProtocol(maxiters, intermediate_save)

# BBO implementation
function learn(protocol::BBOProtocol, learning_problem, p_repr_ig; 
               logging::Bool=false, callback_config::CallbackConfig=CallbackConfig())
    
    callback, loss_history = create_bbo_callback_with_early_termination(
        callback_config, protocol.maxiters, protocol.intermediate_save_function)

    classical_params_array = collect(values(copy(p_repr_ig.p_classical)))

    function classical_loss(x, p)
        p_repr = reconstruct_learning_params_from_array(x, p_repr_ig, learning_problem.model)
        return get_loss(p_repr; learning_problem=learning_problem)
    end

    prob = Optimization.OptimizationProblem(
        classical_loss,
        classical_params_array,
        [1.0, 100.0];
        lb=learning_problem.p_repr_lb,
        ub=learning_problem.p_repr_ub,
    )

    sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=protocol.maxiters)

    final_params = reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)

    convergence_status = if length(loss_history) < protocol.maxiters
        :early_termination
    elseif sol.retcode == :success
        :converged
    else
        :max_iterations
    end

    parameter_history = logging ? nothing : nothing
    
    return create_learning_result("bbo", final_params, loss_history, 
                                 parameter_history, learning_problem; 
                                 convergence_status=convergence_status)
end