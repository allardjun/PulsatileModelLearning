
# Function to merge default configuration with user-provided configuration
function merge_config(junior_config, senior_config)
    merge(junior_config, senior_config)
end

# Seniority of configuration settings: most-senior to least-senior 
# commandline args (highest precedence) > toml > my_notebook_config > default_config (lowest precedence)

function parse_on_times_idx(val)
    val == "all" ? [1,2,3,4,5,6] : [parse(Int, val)]
end

function parse_command_args(args)
    config_updates = Dict()
    
    i = 3
    while i <= length(args)
        if args[i] == "--model_name" && i + 1 <= length(args)
            config_updates["model_name"] = args[i + 1]
            i += 2
        elseif args[i] == "--mask_id" && i + 1 <= length(args)
            config_updates["mask_id"] = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--flexi_dofs" && i + 1 <= length(args)
            config_updates["flexi_dofs"] = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--loss_strategy" && i + 1 <= length(args)
            config_updates["loss_strategy"] = args[i + 1]
            i += 2
        elseif args[i] == "--on_times_idx" && i + 1 <= length(args)
            config_updates["these_on_time_indexes"] = parse_on_times_idx(args[i + 1])
            i += 2
        else
            i += 1
        end
    end
    
    return config_updates
end

function default_config()
    Dict(
        "run_name" => "framboise99",
        "maxiters_bb" => 100000,
        "these_on_time_indexes" => [4],
        "continuous_pulses" => false,
        "mask_id" => 0,
        "loss_strategy" => "masked_normalized",
        # Default optimizer hyperparameters
        "simplex_hyperparams" => Dict(
            # "stopval" => nothing,
            # "xtol_rel" => 1e-6,
            # "xtol_abs" => 1e-6,
            # "constrtol_abs" => 1e-8,
            "initial_step" => nothing
        ),
        "cmaes_hyperparams" => Dict(
            # "c_1" => 0.0,
            # "c_c" => 0.0,
            # "c_mu" => 0.0,
            # "c_sigma" => 0.0,
            # "c_m" => 1.0,
            "sigma0" => -1.0,  # Use -1 for adaptive sigma0 computation
            # "weights" => nothing
        ),
    )
end

function get_config(args, notebook_config)
    config = merge_config(default_config(), notebook_config)
    
    # Load TOML if provided
    if !isempty(args) && endswith(args[1], ".toml")
        toml_config = TOML.parsefile(args[1])
        config = merge_config(config, toml_config)
        
        # Deep merge for nested hyperparameter dictionaries
        if haskey(toml_config, "simplex_hyperparams")
            config["simplex_hyperparams"] = merge(config["simplex_hyperparams"], toml_config["simplex_hyperparams"])
        end
        if haskey(toml_config, "cmaes_hyperparams")
            config["cmaes_hyperparams"] = merge(config["cmaes_hyperparams"], toml_config["cmaes_hyperparams"])
        end
    end
    
    # Override run_name if provided
    length(args) > 1 && (config["run_name"] = args[2])
    
    # Parse command-line flags
    length(args) > 2 && merge!(config, parse_command_args(args))
    
    return config
end

# Helper functions to load optimizer hyperparameters from config
function load_simplex_hyperparams(config)
    params = get(config, "simplex_hyperparams", Dict())
    return (
        stopval = get(params, "stopval", nothing),
        xtol_rel = get(params, "xtol_rel", 1e-6),
        xtol_abs = get(params, "xtol_abs", 1e-6),
        constrtol_abs = get(params, "constrtol_abs", 1e-8),
        initial_step = get(params, "initial_step", nothing)
    )
end

function load_cmaes_hyperparams(config)
    params = get(config, "cmaes_hyperparams", Dict())
    return (
        c_1 = get(params, "c_1", 0.0),
        c_c = get(params, "c_c", 0.0),
        c_mu = get(params, "c_mu", 0.0),
        c_sigma = get(params, "c_sigma", 0.0),
        c_m = get(params, "c_m", 1.0),
        sigma0 = get(params, "sigma0", -1.0),  # Default to adaptive sigma0
        weights = get(params, "weights", nothing)
    )
end
