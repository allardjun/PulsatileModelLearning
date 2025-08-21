
using Model8

# using Makie
# using CairoMakie

using JLD2

using Random
using LibGit2

using Zygote

using SciMLSensitivity

using Infiltrator # for debugging
Infiltrator.toggle_async_check(false)

seed = rand(UInt)#52215207017607922#rand(UInt)
println("Random seed: $seed")
Random.seed!(seed)

my_notebook_config = Dict(
    "run_name" => "gochujang9",
    "model_name" => "ModelF8a",
    "maxiters_bb" => 0,#1000,
    "maxiters_simplex" => 200,#1000,
    "maxiters_cmaes" => 3,
    "maxiters_outer" => 3,
    "these_on_time_indexes" => [6],#[1, 2, 3, 4, 5, 6],#[1,2,3,4,5,6],
    "ig_for_cmaes" => Dict(
        "data_date" => "250708",
        "model_name" => "MyModel8",
        "run_name_base" => "kirsch4_task0a_classical",
        "these_on_time_indexes" => [6],#[6],
    ),
    "flexi_dofs" => 40,
    "continuous_pulses" => false,
    "mask_id" => 0, # 0 will use mask_pairs to generate a mask
    "mask_pairs" => [],#[[1,1],[1,2],[1,3],[1,4]],
    # CMAES hyperparameters
    "cmaes_hyperparams" => Dict(
        # "c_1" => 0.0,
        # "c_c" => 0.0,
        # "c_mu" => 0.0,
        # "c_sigma" => 0.0,
        # "c_m" => 1.0,
        "sigma0" => -1,  # Use -1 for adaptive sigma0 computation
        # "weights" => nothing  # Optional
    ),
    # Simplex hyperparameters
    "simplex_hyperparams" => Dict(
        # "stopval" => 1e-8,
        # "xtol_rel" => 1e-6,
        # "xtol_abs" => 1e-6,
        # "constrtol_abs" => 1e-8,
        # "initial_step" => nothing  # Optional, problem-specific
    ),
)

my_config = Model8.get_config(ARGS, my_notebook_config)

base_path = pwd()  # Repository root for loading experimental data
data_path = Model8.get_experiment_data_path()  # experiments/YYMMDD/data/

this_run_description =
    my_config["run_name"] * "_" * my_config["model_name"] * "_" * join(my_config["these_on_time_indexes"], "_")

println(this_run_description)

# Print the current git hash
repo = LibGit2.GitRepo(".")
commit = LibGit2.head(repo)
hash = LibGit2.GitHash(commit)
println("commit " * string(hash))

@load joinpath(base_path, "data/Harris_data_CD69.jld2") data

readout_protein = "CD69"
on_times_all = data[readout_protein]["on_times"]
off_times_all = data[readout_protein]["off_times"]
c24_data_all = data[readout_protein]["c24"]

on_times = on_times_all[my_config["these_on_time_indexes"]]
off_times = off_times_all[my_config["these_on_time_indexes"]]
c24_data = transpose(hcat(c24_data_all[my_config["these_on_time_indexes"]]...))

model = Model8.create_model(my_config["model_name"]; flexi_dofs=my_config["flexi_dofs"])

# construct mask that excludes certain on_time,off_time pairs from SSR, either for outlier rejection or k-fold cross-validation
mask = Model8.generate_mask(my_config, c24_data)

learning_problem = LearningProblem(;
    on_times=on_times,
    off_times=off_times,
    c24_data=c24_data,
    p_repr_lb=Model8.represent(model.p_derepresented_lowerbounds, model),
    p_repr_ub=Model8.represent(model.p_derepresented_upperbounds, model),
    model=model,
    continuous_pulses=my_config["continuous_pulses"],
    mask = mask,
)

params_repr = deepcopy(model.params_repr_ig)
# p_repr = deepcopy(Model8.params_repr_ig)

## ------ test loss function on initial guess ------ ##

println("Time for one call to get_loss:")

@time loss_ig = Model8.get_loss(params_repr; learning_problem=learning_problem)

@time loss_ig = Model8.get_loss(params_repr; learning_problem=learning_problem)


## ------ cmaes optimization ------ ##

if my_config["maxiters_outer"] > 0
    ig_for_corduroy = params_repr # uses best from differential evolution
    # initial guess
    if haskey(my_config, "ig_for_cmaes") && !isnothing(my_config["ig_for_cmaes"]) && 
       haskey(my_config["ig_for_cmaes"], "data_date") && !isempty(my_config["ig_for_cmaes"]["data_date"])
        
        # Handle special signal value -1: use same on_times_indexes as target dataset
        ig_on_times_indexes = if my_config["ig_for_cmaes"]["these_on_time_indexes"] == [-1]
            my_config["these_on_time_indexes"]  # Use target dataset
        else
            my_config["ig_for_cmaes"]["these_on_time_indexes"]  # Use specified dataset
        end
        
        # Construct pattern to match any job_id for the target dataset
        dataset_desc = length(ig_on_times_indexes) == 6 ? "1_2_3_4_5_6" : string(ig_on_times_indexes[1])
        
        # Pattern matching approach - find file with or without job_id using regex
        file_prefix = my_config["ig_for_cmaes"]["run_name_base"] 
        file_suffix = "_" * my_config["ig_for_cmaes"]["model_name"] * "_" * dataset_desc * "_simplex.jdl2"
        
        # Create regex that matches both patterns:
        # 1. run_name_base_job[INTEGER]_model_name_dataset_desc_simplex.jdl2
        # 2. run_name_base_model_name_dataset_desc_simplex.jdl2
        file_regex = Regex("^" * escape_string(file_prefix) * "(_job\\d+)?" * escape_string(file_suffix) * "\$")
        
        # Find matching files in the data directory
        data_path_for_search = joinpath(base_path, "experiments", my_config["ig_for_cmaes"]["data_date"], "data")
        
        if !isdir(data_path_for_search)
            error("Data directory does not exist: $data_path_for_search")
        end
        
        all_files = readdir(data_path_for_search)
        matching_files = filter(f -> occursin(file_regex, f), all_files)
        
        if length(matching_files) == 1
            data_file_name = matching_files[1]
        elseif length(matching_files) == 0
            error("No files found matching pattern: $(file_prefix)(_job[INTEGER])?$(file_suffix) in $data_path_for_search")
        else
            error("Multiple files found matching pattern: $(file_prefix)(_job[INTEGER])?$(file_suffix) in $data_path_for_search: $matching_files")
        end
        println("Loading initial guess from: ", data_file_name)
        
        # Use unified loading interface
        source_result, source_config, source_learning_problem = Model8.load_learning_result(
            joinpath(base_path, "experiments", my_config["ig_for_cmaes"]["data_date"], "data", data_file_name)
        )
        
        best_p_repr = source_result["parameters"]

        println("Initial guess loss, according to source model:")
        @show best_p_repr
        @show Model8.get_loss(best_p_repr; learning_problem=source_learning_problem)
        @show typeof(source_learning_problem.model)

        bounds_violations = Model8.check_parameter_bounds(best_p_repr, source_learning_problem.model, "source ig in source model", "source ig")


        # Use generic conversion system - automatically detects source model from config
        source_model = source_config["model_name"]
        dest_model = my_config["model_name"]
        pretrained_ig_for_cmaes, loss_pretraining = Model8.convert_model(source_model, dest_model, best_p_repr; flexi_dofs=my_config["flexi_dofs"])

        @show pretrained_ig_for_cmaes

        println("Initial guess loss, pre-trained into destination model:")
        @show Model8.get_loss(pretrained_ig_for_cmaes; learning_problem=learning_problem)
        @show typeof(learning_problem.model)

        bounds_violations = Model8.check_parameter_bounds(pretrained_ig_for_cmaes, learning_problem.model, "source ig in destination model", "source ig")


        @save joinpath(data_path, this_run_description * "_pretraining.jdl2") pretrained_ig_for_cmaes loss_pretraining my_config learning_problem

        # @show pretrained_ig_for_cmaes

        # pretrained_ig_for_cmaes
        ig_for_corduroy = pretrained_ig_for_cmaes

        # error("Stop here.") # for debugging

    end

    ## ---- non-logging mode ---- #

    # Initialize parameters and loss history
    current_params = deepcopy(ig_for_corduroy)
    corduroy_loss_history = []

    for round in 1:my_config["maxiters_outer"]
        println("\nStarting round $round of $(my_config["maxiters_outer"])")

        global current_params
        
        # Declare local variables to avoid scoping warnings
        local bounds_violations, cmaes_protocol, result, cmaes_loss_history
        local simplex_protocol, simplex_loss_history

        ## ------ CMAES optimization ------ ##

        # Debug: Parameter transfer TO cmaes
        println("Parameters passed TO CMA-ES in round $round")
        println("  Loss before CMA-ES: $(Model8.get_loss(current_params; learning_problem=learning_problem))")
        println("  Classical param sum: $(sum(abs.(current_params.p_classical)))")
        println("  Flexi1 param sum: $(sum(abs.(current_params.flex1_params)))")
        if haskey(current_params, :flex2_params)
            println("  Flexi2 param sum: $(sum(abs.(current_params.flex2_params)))")
        end
        bounds_violations = Model8.check_parameter_bounds(current_params, learning_problem.model, "Pre-CMAES-R$round", "before_cmaes")

        # Load CMAES hyperparameters from config
        cmaes_params = Model8.load_cmaes_hyperparams(my_config)
        cmaes_protocol = Model8.CMAESProtocol(
            maxiters=my_config["maxiters_cmaes"];
            cmaes_params...
        )
        @time result = Model8.learn(cmaes_protocol, learning_problem, current_params)
        current_params, cmaes_loss_history = result["parameters"], result["loss_history"]

        # Debug: Parameter transfer FROM cmaes
        println("Parameters received FROM CMA-ES in round $round")
        println("  Loss after CMA-ES: $(Model8.get_loss(current_params; learning_problem=learning_problem))")
        println("  Classical param sum: $(sum(abs.(current_params.p_classical)))")
        println("  Flexi1 param sum: $(sum(abs.(current_params.flex1_params)))")
        if haskey(current_params, :flex2_params)
            println("  Flexi2 param sum: $(sum(abs.(current_params.flex2_params)))")
        end
        bounds_violations = Model8.check_parameter_bounds(current_params, learning_problem.model, "Post-CMAES-R$round", "after_cmaes")

        #debugging
        # @show current_params
        # @show Model8.get_loss(current_params; learning_problem=learning_problem) 

        push!(corduroy_loss_history, (cmaes_loss_history, "cmaes"))

        ## ------ Nelder-Mead optimization ------ ##

        # Debug: Parameter transfer TO simplex
        println("Parameters passed TO Simplex in round $round")
        println("  Loss before Simplex: $(Model8.get_loss(current_params; learning_problem=learning_problem))")
        println("  Classical param sum: $(sum(abs.(current_params.p_classical)))")
        println("  Flexi1 param sum: $(sum(abs.(current_params.flex1_params)))")
        if haskey(current_params, :flex2_params)
            println("  Flexi2 param sum: $(sum(abs.(current_params.flex2_params)))")
        end
        bounds_violations = Model8.check_parameter_bounds(current_params, learning_problem.model, "Pre-Simplex-R$round", "before_simplex")

        # Load simplex hyperparameters from config
        simplex_params = Model8.load_simplex_hyperparams(my_config)
        simplex_protocol = Model8.SimplexProtocol(
            maxiters=my_config["maxiters_simplex"];
            simplex_params...
        )
        @time result = Model8.learn(simplex_protocol, learning_problem, current_params)
        current_params, simplex_loss_history = result["parameters"], result["loss_history"]

        # Debug: Parameter transfer FROM simplex
        println("Parameters received FROM Simplex in round $round")
        println("  Loss after Simplex: $(Model8.get_loss(current_params; learning_problem=learning_problem))")
        println("  Classical param sum: $(sum(abs.(current_params.p_classical)))")
        println("  Flexi1 param sum: $(sum(abs.(current_params.flex1_params)))")
        if haskey(current_params, :flex2_params)
            println("  Flexi2 param sum: $(sum(abs.(current_params.flex2_params)))")
        end
        bounds_violations = Model8.check_parameter_bounds(current_params, learning_problem.model, "Post-Simplex-R$round", "after_simplex")

        #debugging
        # @show current_params
        # @show Model8.get_loss(current_params; learning_problem=learning_problem)

        push!(corduroy_loss_history, (simplex_loss_history, "simplex"))

        println(
            "\nRound $round complete. Current loss: $(Model8.get_loss(current_params; learning_problem=learning_problem))",
        )

    end

    ## ---- --- #

    best_p_repr = current_params

    # Create a special corduroy result with composite loss history
    final_loss = Model8.get_loss(best_p_repr; learning_problem=learning_problem)
    total_iterations = sum(length(hist[1]) for hist in corduroy_loss_history)
    
    corduroy_result = Model8.create_learning_result_dict(
        best_p_repr,
        Float32[],  # Composite loss history stored in metadata
        nothing,
        final_loss,
        total_iterations,
        :completed,
        "corduroy",
        Dict("corduroy_loss_history" => corduroy_loss_history)
    )
    
    Model8.save_learning_result(joinpath(data_path, this_run_description * "_corduroy.jdl2"), corduroy_result, my_config, learning_problem)
end
