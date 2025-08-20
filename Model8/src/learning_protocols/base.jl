# Base Learning Protocol Components
# Contains abstract types, constants, callbacks, I/O, and factory functions

using ComponentArrays
using Dates

# Abstract base types
abstract type LearningProtocol end
abstract type DerivativeFreeProtocol <: LearningProtocol end

# Learning constants
struct LearningConstants
    qdrms_divisor::Float64
    convergence_threshold::Float64
    early_check_fraction::Float64
    min_early_iterations::Int
end

const DEFAULT_LEARNING_CONSTANTS = LearningConstants(14.0, 1/1000, 1/6, 20000)

# Learning result structure - using Dict for simplicity and compatibility
const LearningResult = Dict{String, Any}

# Helper function to create a LearningResult
function create_learning_result_dict(parameters, loss_history, parameter_history, final_loss, 
                                    iterations, convergence_status, protocol_name, metadata)
    return Dict{String, Any}(
        "parameters" => parameters,
        "loss_history" => loss_history,
        "parameter_history" => parameter_history,
        "final_loss" => final_loss,
        "iterations" => iterations,
        "convergence_status" => convergence_status,
        "protocol_name" => protocol_name,
        "metadata" => metadata
    )
end

# Progress callback configuration
struct CallbackConfig
    print_frequency::Int
    save_parameters::Bool
    verbose::Bool
    constants::LearningConstants
end

CallbackConfig(; print_frequency=1, save_parameters=false, verbose=true, 
               constants=DEFAULT_LEARNING_CONSTANTS) = 
    CallbackConfig(print_frequency, save_parameters, verbose, constants)

# Standardized callback functions
function create_standard_callback(protocol_name::String, config::CallbackConfig)
    loss_history = Float32[]
    parameter_history = config.save_parameters ? [] : nothing
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if config.verbose && current_iter % config.print_frequency == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In $protocol_name, iteration $current_iter: loss=$lossval, qdrms=$qdrms at $(now())")
            flush(stdout)
        end
        
        if config.save_parameters
            push!(parameter_history, deepcopy(p))
        end
        
        return false
    end
    
    return callback, loss_history, parameter_history
end

function create_bbo_callback_with_early_termination(config::CallbackConfig, maxiters::Int, 
                                                  intermediate_save=nothing)
    loss_history = Float32[]
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if config.verbose && current_iter % 1000 == 0
            qdrms = sqrt(lossval / config.constants.qdrms_divisor)
            println("In bb, Current loss after $current_iter iterations: $lossval, qd rms=$qdrms at $(now())")
            flush(stdout)
        end

        if !isnothing(intermediate_save) && current_iter % 10000 == 0
            # Note: intermediate save would need specific implementation
        end
        
        early_check_point = max(ceil(Int, maxiters * config.constants.early_check_fraction), 
                               config.constants.min_early_iterations)
        if current_iter >= early_check_point
            halfway_point = current_iter - ceil(Int, current_iter / 2)
            previous_loss = loss_history[halfway_point]
            current_loss = loss_history[end]
            relative_change = abs(current_loss - previous_loss) / current_loss
            
            if relative_change < config.constants.convergence_threshold
                println("Early termination at iteration $current_iter: Loss stabilized (change: $relative_change).")
                return true
            end
        end

        return false
    end
    
    return callback, loss_history
end

# Helper function to create learning result
function create_learning_result(protocol_name::String, final_params, loss_history, 
                               parameter_history, learning_problem; 
                               convergence_status=:completed, metadata=Dict{String, Any}())
    final_loss = get_loss(final_params; learning_problem=learning_problem)
    iterations = length(loss_history)
    
    println("After $protocol_name: loss=$final_loss")
    
    return create_learning_result_dict(
        final_params,
        loss_history,
        parameter_history,
        final_loss,
        iterations,
        convergence_status,
        protocol_name,
        metadata
    )
end

# Save and load functions for learning results
function save_learning_result(filepath::String, result::LearningResult, config, learning_problem)
    # Create a comprehensive save that includes everything needed for analysis
    save_data = Dict(
        "result" => result,
        "config" => config,
        "learning_problem" => learning_problem,
        # Backward compatibility fields
        "best_p_repr" => result["parameters"],
        "$(result["protocol_name"])_loss_history" => result["loss_history"],
        "my_config" => config,
    )
    
    # Add parameter history if available
    if !isnothing(result["parameter_history"])
        save_data["$(result["protocol_name"])params_history"] = result["parameter_history"]
    end
    
    # Save with JLD2 - avoid splatting to prevent Symbol/String key issues
    jldopen(filepath, "w") do file
        for (key, value) in save_data
            file[key] = value
        end
    end
end

function load_learning_result(filepath::String)::Tuple{LearningResult, Any, Any}
    data = load(filepath)
    
    # Try to load new format first (check both Symbol and String keys for compatibility)
    if haskey(data, :result) || haskey(data, "result")
        result = get(data, :result, get(data, "result", nothing))
        config = get(data, :config, get(data, "config", nothing))
        learning_problem = get(data, :learning_problem, get(data, "learning_problem", nothing))
        return result, config, learning_problem
    end
    
    # Fallback to legacy format - reconstruct LearningResult from individual fields
    config = get(data, :my_config, get(data, "my_config", nothing))
    learning_problem = get(data, :learning_problem, get(data, "learning_problem", nothing))
    best_p_repr = get(data, :best_p_repr, get(data, "best_p_repr", nothing))
    
    # Determine protocol name from available loss history
    protocol_name = "unknown"
    loss_history = Float32[]
    parameter_history = nothing
    
    for key in keys(data)
        key_str = string(key)  # Convert Symbol to String for consistent checking
        if endswith(key_str, "_loss_history")
            protocol_name = replace(key_str, "_loss_history" => "")
            loss_history = data[key]
            break
        end
    end
    
    # Handle special cases
    if protocol_name == "cmaes" && length(loss_history) > 1
        loss_history = loss_history[2:end]  # Remove first element as in original code
    elseif haskey(data, :corduroy_loss_history) || haskey(data, "corduroy_loss_history")
        protocol_name = "corduroy" 
        loss_history = Float32[]  # Composite loss stored in metadata
    end
    
    final_loss = get_loss(best_p_repr; learning_problem=learning_problem)
    
    result = create_learning_result_dict(
        best_p_repr,
        loss_history,
        parameter_history,
        final_loss,
        length(loss_history),
        :completed,  # Unknown status for legacy files
        protocol_name,
        Dict{String, Any}()
    )
    
    return result, config, learning_problem
end

# Main learning interface
function learn(protocol::LearningProtocol, learning_problem, initial_params; 
               maxiters::Int=1000, logging::Bool=false, 
               callback_config::CallbackConfig=CallbackConfig())
    error("learn() must be implemented for protocol type $(typeof(protocol))")
end

# Protocol factory function
function create_protocol(protocol_name::String; kwargs...)
    if protocol_name == "bbo"
        return BBOProtocol(; kwargs...)
    elseif protocol_name == "cmaes"
        return CMAESProtocol(; kwargs...)
    elseif protocol_name == "simplex"
        return SimplexProtocol(; kwargs...)
    else
        error("Unknown protocol: $protocol_name. Available protocols: bbo, cmaes, simplex")
    end
end

# Convenience function for backward compatibility
function learn_with_protocol(protocol_name::String, learning_problem, initial_params; 
                           maxiters::Int=1000, logging::Bool=false, 
                           callback_config::CallbackConfig=CallbackConfig(),
                           protocol_kwargs...)
    protocol = create_protocol(protocol_name; maxiters=maxiters, protocol_kwargs...)
    return learn(protocol, learning_problem, initial_params; 
                logging=logging, callback_config=callback_config)
end