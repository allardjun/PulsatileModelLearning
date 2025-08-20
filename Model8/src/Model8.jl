module Model8

# using JunTools
using XLSX
using Statistics
using JLD2
using DataStructures
using ComponentArrays
using OrdinaryDiffEq

using Makie

using Parameters
using Optimization
using OptimizationBBO # This contains BBO Differential Evolution
using OptimizationMetaheuristics

using OptimizationOptimJL # This contains NelderMead and BFGS
using OptimizationOptimisers # This contains ADAM
using OptimizationNLopt # This contains NLopt algorithms including LN_SBPLX
import NLopt # For direct NLopt debugging in simplex.jl
using Optim
# using OptimizationGCMAES
using OptimizationEvolutionary

using Zygote # used in pre-training flexis after a classical run

# using ForwardDiff
# using Enzyme
using SciMLSensitivity

# using LineSearches

using Dates

using TOML

using InteractiveUtils
using Logging
using Infiltrator

using Sundials
using LinearAlgebra

using Random

include("FlexiFunctions.jl")
include("ModelConversion.jl")

using .FlexiFunctions
using .ModelConversion

base_path = ""
# base_path = JunTools.get_base_path("TCRPulsing")

#export base_path

export LearningProblem

# Export model factory functions
export create_model, register_model!, unregister_model!
export list_models, get_model_info, print_available_models

# Export new unified learning protocol interface
export LearningProtocol, LearningResult, CallbackConfig, LearningConstants
export BBOProtocol, CMAESProtocol, SimplexProtocol
export learn, create_protocol, learn_with_protocol
export save_learning_result, load_learning_result

# Export model conversion functions
export convert_model, convert_from_saved_result, list_supported_conversions

# Legacy functions removed - all callers updated to use new interface

abstract type AbstractModelTrait end # stores only which model
abstract type AbstractModel end # stores which model, and also parameter values

# Include shared model functions after AbstractModel is defined
include("shared_model_functions.jl")

@with_kw struct LearningProblem{M<:AbstractModel}
    on_times::Vector{Float32}
    off_times::Vector{Any}
    c24_data::Matrix{Float64}
    p_repr_lb::ComponentArray{Float64}
    p_repr_ub::ComponentArray{Float64}
    model::M
    continuous_pulses::Bool
    mask::Matrix{Bool}
    loss_strategy::String = "masked_normalized"
end

include("get_data.jl")
include("define_input_functions.jl")

# Include shared model functions first (needed by individual model files)
# Functions are defined at module level, so they'll be available to model files

include("models/define_model5.jl")
include("models/define_model6.jl")
include("models/define_model7.jl")

include("models/define_model8.jl")

include("models/define_modelF6.jl")
include("models/define_modelF7.jl")
include("models/define_modelF8.jl")

include("models/define_modelF8a.jl")
include("models/define_modelF8b.jl")


# Model Factory Registry
## -----------------

# Model registration system
struct ModelSpec
    make_function::Function
    supports_flexi_dofs::Bool
    description::String
end

# Global model registry
const MODEL_REGISTRY = Dict{String, ModelSpec}()

# Registration functions
function register_model!(name::String, make_func::Function; supports_flexi_dofs::Bool=false, description::String="")
    MODEL_REGISTRY[name] = ModelSpec(make_func, supports_flexi_dofs, description)
    return nothing
end

function unregister_model!(name::String)
    delete!(MODEL_REGISTRY, name)
    return nothing
end

# Factory function
function create_model(model_name::String; flexi_dofs::Int=40)
    haskey(MODEL_REGISTRY, model_name) || error("Model '$(model_name)' not found. Available models: $(join(keys(MODEL_REGISTRY), ", "))")
    
    spec = MODEL_REGISTRY[model_name]
    
    if spec.supports_flexi_dofs
        return spec.make_function(; flexi_dofs=flexi_dofs)
    else
        return spec.make_function()
    end
end

# Utility functions
function list_models()::Vector{String}
    return collect(keys(MODEL_REGISTRY))
end

function get_model_info(model_name::String)::ModelSpec
    haskey(MODEL_REGISTRY, model_name) || error("Model '$(model_name)' not found")
    return MODEL_REGISTRY[model_name]
end

function print_available_models()
    println("Available Models:")
    println("=================")
    for (name, spec) in sort(collect(MODEL_REGISTRY), by=first)
        flexi_support = spec.supports_flexi_dofs ? " (supports flexi_dofs)" : ""
        desc = isempty(spec.description) ? "" : " - $(spec.description)"
        println("  $(name)$(flexi_support)$(desc)")
    end
end

# Register available models
function _register_default_models!()
    register_model!("MyModel8", make_MyModel8; description="DI-IFFL model")
    register_model!("Model5", make_Model5; description="Linear architecture with production-promotion")
    register_model!("Model6", make_Model6; description="Linear with decay-inhibition")
    register_model!("Model7", make_Model7; description="Incoherent feedforward")
    register_model!("ModelF6", make_ModelF6; supports_flexi_dofs=true, description="Linear with decay-inhibition and flexi function")
    register_model!("ModelF7", make_ModelF7; supports_flexi_dofs=true, description="Linear architecture with production-promotion and flexi function")
    register_model!("ModelF8", make_ModelF8; supports_flexi_dofs=true, description="DI-IFFL with flexi functions in m-PP and m-DI")
    register_model!("ModelF8a", make_ModelF8a; supports_flexi_dofs=true, description="DI-IFFL with flexi functions in m-PP and m-DI (copy of ModelF8)")
    register_model!("ModelF8b", make_ModelF8b; supports_flexi_dofs=true, description="DI-IFFL with flexi functions in m-PP and m-DI (copy of ModelF8)")
end

# Initialize the registry when module loads
_register_default_models!()

## -----------------

include("get_freq_response.jl")
include("get_loss.jl")

# New unified learning protocols (split into separate files)
include("learning_protocols/base.jl")
include("learning_protocols/bbo.jl")
include("learning_protocols/cmaes.jl")
include("learning_protocols/simplex.jl")

include("config_tools.jl")
include("get_metrics.jl")

include("generate_masks.jl")
include("plotting.jl")
include("timeseries_analysis.jl")

end # module Model8
