# Shared model functions to reduce duplication across model definitions
# This module contains common functions that are identical or follow templates across all models

# Base types for dispatch
abstract type AbstractClassicalModel <: AbstractModel end  
abstract type AbstractFlexiModel <: AbstractModel end

# Generic represent function (100% identical across all models)
function represent(p_derepresented, model::AbstractModel)
    return represent_on_type(p_derepresented, typeof(model))
end

# Template-based derepresent_all functions
function derepresent_all(p_repr_all, model::AbstractClassicalModel)
    return ComponentArray(p_classical=derepresent(p_repr_all.p_classical, model))
end

function derepresent_all(p_repr_all, model::AbstractFlexiModel)  
    result_dict = Dict{Symbol, Any}(
        :p_classical => derepresent(p_repr_all.p_classical, model)
    )
    
    # Add flexi params that actually exist in the parameter structure
    if haskey(p_repr_all, :flex1_params)
        result_dict[:flex1_params] = p_repr_all.flex1_params
    end
    
    if haskey(p_repr_all, :flex2_params)
        result_dict[:flex2_params] = p_repr_all.flex2_params
    end
    
    return ComponentArray(result_dict)
end

# Template-based reconstruction functions for classical models
function reconstruct_learning_params_from_array(classical_params_array, p_repr_all, model::AbstractClassicalModel)
    keys_list = collect(keys(p_repr_all.p_classical))
    p_repr_classical = ComponentArray{Float64}(;
        (keys_list[i] => classical_params_array[i] for i in eachindex(classical_params_array))...
    )
    return ComponentArray(p_classical=p_repr_classical)
end

# Template-based reconstruction functions for flexi models  
function reconstruct_learning_params_from_array(classical_params_array, p_repr_all, model::AbstractFlexiModel)
    keys_list = collect(keys(p_repr_all.p_classical))
    p_repr_classical = ComponentArray{Float64}(;
        (keys_list[i] => classical_params_array[i] for i in eachindex(classical_params_array))...
    )
    
    result_dict = Dict{Symbol, Any}(:p_classical => p_repr_classical)
    
    # Add flexi params that actually exist in the parameter structure
    if haskey(p_repr_all, :flex1_params)
        result_dict[:flex1_params] = p_repr_all.flex1_params
    end
    
    if haskey(p_repr_all, :flex2_params)
        result_dict[:flex2_params] = p_repr_all.flex2_params
    end
    
    return ComponentArray(result_dict)
end

function reconstruct_learning_params(classical_params, p_repr_all, model::AbstractClassicalModel)
    return ComponentArray(p_classical=classical_params)
end

function reconstruct_learning_params(classical_params, p_repr_all, model::AbstractFlexiModel)
    new_p_repr_all = deepcopy(p_repr_all)  # Preserve the entire current parameter structure
    new_p_repr_all.p_classical = classical_params  # Update only the classical parameters
    return new_p_repr_all
end

# Parameter bounds checking function - moved from analyze_task0b_freq.jl for reusability
function check_parameter_bounds(best_p, model, model_name, dataset_desc)
    """Check if optimized parameters are within the specified bounds"""
    
    # Get derepresented parameters (biophysical values)
    p_derepresented = derepresent_all(best_p, model)
    
    # Get model bounds
    lower_bounds = model.p_derepresented_lowerbounds
    upper_bounds = model.p_derepresented_upperbounds
    
    violations = []
    
    # Check each parameter
    for param_name in keys(p_derepresented.p_classical)
        value = p_derepresented.p_classical[param_name]
        lower = lower_bounds[param_name]
        upper = upper_bounds[param_name]
        
        if value < lower
            push!(violations, (param=param_name, value=value, bound=lower, type="lower"))
        elseif value > upper
            push!(violations, (param=param_name, value=value, bound=upper, type="upper"))
        end
    end
    
    # Report results
    if isempty(violations)
        println("✓ $model_name ($dataset_desc): All parameters within bounds")
    else
        println("✗ $model_name ($dataset_desc): $(length(violations)) bound violations:")
        for v in violations
            println("  $(v.param): $(v.value) violates $(v.type) bound $(v.bound)")
        end
    end
    
    return violations
end