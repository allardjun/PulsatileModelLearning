"""
Generic model conversion system to replace manual model-specific conversion functions.
Converts classical models (Model6, Model7, MyModel8) to their flexible counterparts (ModelF6, ModelF7, ModelF8).
"""
module ModelConversion

using ComponentArrays
using ..FlexiFunctions

export convert_model, convert_from_saved_result, list_supported_conversions

"""
Configuration for converting between model types.
Defines which parameters need transformation and how flexi functions should be initialized.
"""
struct ConversionMapping
    # Parameters to copy directly from source to destination
    direct_params::Vector{Symbol}
    
    # Parameters that need exp() transformation for flexi initialization
    beta_params::Vector{Symbol}
    
    # Parameters that need ^2 transformation for flexi initialization  
    n_params::Vector{Symbol}
    
    # Names of flexi parameter arrays in destination model
    flexi_param_names::Vector{Symbol}
    
    # Special initialization function (optional)
    special_init::Union{Nothing, Function}
end

# """
# Special initialization function for ModelF8b conversion (legacy - no longer used).
# """
# function _special_model8b_init(flexi_dofs::Int, beta_params::Dict, n_params::Dict)
#     flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs; 
#                                                     beta=beta_params[:beta_mw], 
#                                                     n=n_params[:nwm])
#     
#     # Special initialization for flex2_params in ModelF8b
#     special_beta = 1 + (beta_params[:beta_ma])^n_params[:nam] 
#     flex2_params = FlexiFunctions.generate_flexi_ig(flexi_dofs; beta=special_beta, n=1.0)
#     
#     return [flex1_params, flex2_params]
# end

"""
Registry of all supported model conversions.
Maps (source_model, dest_model) pairs to their conversion configurations.
"""
const CONVERSION_REGISTRY = Dict{Tuple{String,String}, ConversionMapping}(
    # Model6 -> ModelF6: Single flexi function (flex1_params) for beta_ma/nam
    ("Model6", "ModelF6") => ConversionMapping(
        [:tau_a, :tau_m, :tau_w, :tau_c],
        [:beta_ma],
        [:nam], 
        [:flex1_params],
        nothing
    ),
    
    # Model7 -> ModelF7: Single flexi function (flex1_params) for beta_mw/nwm  
    ("Model7", "ModelF7") => ConversionMapping(
        [:tau_a, :tau_m, :tau_w, :tau_c],
        [:beta_mw],
        [:nwm],
        [:flex1_params],
        nothing
    ),
    
    # Model8 -> ModelF8: Dual flexi functions
    ("MyModel8", "ModelF8") => ConversionMapping(
        [:tau_a, :tau_m, :tau_w, :tau_c, :nwm, :nam],
        [:beta_mw, :beta_ma],
        [:nwm, :nam],
        [:flex1_params, :flex2_params],
        nothing
    ),
    
    # Model8 -> ModelF8a: Single flexi function in IFFL, 
    # flex1 for IFFL requires beta_mw and nwm
    # classical function for DI requires beta_ma and nam
    ("MyModel8", "ModelF8a") => ConversionMapping(
        [:tau_a, :tau_m, :tau_w, :tau_c, :beta_ma, :nwm, :nam],
        [:beta_mw],
        [:nwm],
        [:flex1_params],
        nothing
    ),
    
    # Model8 -> ModelF8b: Single flexi function for DI
    # flex1 for DI requires beta_ma and nam
    # classical function for IFFL requires beta_mw and nwm
    ("MyModel8", "ModelF8b") => ConversionMapping(
        [:tau_a, :tau_m, :tau_w, :tau_c, :beta_mw, :nwm, :nam],
        [:beta_ma],
        [:nam],
        [:flex1_params],
        nothing
    )
)

"""
Generic conversion function that replaces all model-specific conversion functions.

# Arguments
- `source_model::String`: Name of source model (e.g., "MyModel8")
- `dest_model::String`: Name of destination model (e.g., "ModelF8") 
- `p_repr_source_model::ComponentArray`: Parameters from trained classical model
- `flexi_dofs::Int=20`: Number of degrees of freedom for flexi functions

# Returns
- `Tuple{ComponentArray, Nothing}`: (converted_parameters, loss_history)
  where loss_history is always nothing for compatibility with existing code
"""
function convert_model(source_model::String, dest_model::String, 
                      p_repr_source_model::ComponentArray; flexi_dofs::Int=20)
    
    println("Hello from convert_model.jl: converting $source_model to $dest_model")

    # Look up conversion mapping
    mapping_key = (source_model, dest_model)
    if !haskey(CONVERSION_REGISTRY, mapping_key)
        error("Unsupported conversion: $source_model -> $dest_model")
    end
    
    mapping = CONVERSION_REGISTRY[mapping_key]
    
    # Step 1: Extract and copy direct parameters
    p_classical_dict = Dict{Symbol, Any}()
    for param in mapping.direct_params
        if haskey(p_repr_source_model.p_classical, param)
            p_classical_dict[param] = p_repr_source_model.p_classical[param]
        end
    end
    p_classical_dest = ComponentArray(p_classical_dict)
    
    # Step 2: Transform beta and n parameters for flexi initialization
    beta_params = Dict{Symbol, Float64}()
    n_params = Dict{Symbol, Float64}()
    
    for param in mapping.beta_params
        if haskey(p_repr_source_model.p_classical, param)
            beta_params[param] = exp(p_repr_source_model.p_classical[param])
        end
    end
    
    for param in mapping.n_params
        if haskey(p_repr_source_model.p_classical, param)
            n_params[param] = (p_repr_source_model.p_classical[param])^2
        end
    end
    
    # Step 3: Initialize flexi parameters
    flexi_dict = Dict{Symbol, Any}()
    
    if mapping.special_init !== nothing
        # Use special initialization function
        flexi_param_values = mapping.special_init(flexi_dofs, beta_params, n_params)
        for (i, param_name) in enumerate(mapping.flexi_param_names)
            flexi_dict[param_name] = flexi_param_values[i]
        end
    else
        # Standard initialization
        if length(mapping.flexi_param_names) == 1
            # Single flexi function
            param_name = mapping.flexi_param_names[1]
            beta_key = mapping.beta_params[1]
            n_key = mapping.n_params[1]
            
            flexi_params = FlexiFunctions.generate_flexi_ig(flexi_dofs; 
                                                          beta=beta_params[beta_key], 
                                                          n=n_params[n_key])
            flexi_dict[param_name] = flexi_params
            
        elseif length(mapping.flexi_param_names) == 2
            # Dual flexi functions
            flex1_params = FlexiFunctions.generate_flexi_ig(flexi_dofs; 
                                                          beta=beta_params[:beta_mw], 
                                                          n=n_params[:nwm])
            flex2_params = FlexiFunctions.generate_flexi_ig(flexi_dofs; 
                                                          beta=beta_params[:beta_ma], 
                                                          n=n_params[:nam]) 
            
            flexi_dict[:flex1_params] = flex1_params
            flexi_dict[:flex2_params] = flex2_params
        else
            error("Unsupported number of flexi functions: $(length(mapping.flexi_param_names))")
        end
    end
    
    # Step 4: Combine into final parameter structure
    final_dict = Dict{Symbol, Any}(:p_classical => p_classical_dest)
    merge!(final_dict, flexi_dict)
    p_repr_dest_model = ComponentArray(final_dict)
    
    return (p_repr_dest_model, nothing)
end

"""
Convenience function that automatically detects source model type from saved classical learning result
and converts to specified destination model.

# Arguments  
- `classical_result_path::String`: Path to saved classical learning result (JLD2 file)
- `dest_model::String`: Target flexible model name (e.g., "ModelF8")
- `flexi_dofs::Int=20`: Number of degrees of freedom for flexi functions

# Returns
- `Tuple{ComponentArray, Nothing}`: (converted_parameters, loss_history)
"""
function convert_from_saved_result(classical_result_path::String, dest_model::String; flexi_dofs::Int=20)
    # Load the classical learning result
    result, config, _ = load_learning_result(classical_result_path)
    
    # Extract source model name from config
    source_model = config["model_name"]
    best_params = result["parameters"]
    
    return convert_model(source_model, dest_model, best_params; flexi_dofs=flexi_dofs)
end

"""
List all supported conversions in the registry.
"""
function list_supported_conversions()
    println("Supported model conversions:")
    for (source, dest) in keys(CONVERSION_REGISTRY)
        println("  $source -> $dest")
    end
end

end # module