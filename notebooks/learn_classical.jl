
using Model8
using JunTools

# using Makie
# using CairoMakie

using JLD2

using Random
using LibGit2
using Dates

# using Zygote
# using SciMLSensitivity

seed = rand(UInt)#52215207017607922#rand(UInt)
println("Random seed: $seed")
Random.seed!(seed)

my_notebook_config = Dict(
    "run_name" => "chimichurri4",
    "model_name" => "Model7",
    "maxiters_bb" => 2000,#1000,
    "maxiters_simplex" => 200,#1000,
    "these_on_time_indexes" => [6],#[1,2,3,4,5,6],#[1],
    # "ig_for_simplex" => Dict(
    #     "data_date" => "250121", 
    #     "run_name" => "falafel1",
    #     ),
    "continuous_pulses" => false,
    "mask_id" => 0, # 0 will use mask_pairs to generate a mask
    "mask_pairs" => [[1,1],[1,2],[1,3],[1,4]],
)

my_config = Model8.get_config(ARGS, my_notebook_config)

base_path = JunTools.get_base_path("TCRPulsing")
data_path = JunTools.get_data_path("TCRPulsing")

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

# select a subset of on_times
on_times = on_times_all[my_config["these_on_time_indexes"]]
off_times = off_times_all[my_config["these_on_time_indexes"]]
c24_data = transpose(hcat(c24_data_all[my_config["these_on_time_indexes"]]...))

# construct learning_problem context object
model = Model8.create_model(my_config["model_name"])

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
    loss_strategy=my_config["loss_strategy"],
)

# unit test for mask_pairs
println("unit test mask_pairs:")
@show my_config["mask_pairs"]
# @show learning_problem.c24_data[learning_problem.mask]
@show size(learning_problem.c24_data[learning_problem.mask])

params_repr = deepcopy(model.params_repr_ig)

##

println(my_config)

## ------ test loss function on initial guess ------ ##

println("Time for one call to get_loss:")

@time loss_ig = Model8.get_loss(params_repr; learning_problem=learning_problem)
@time loss_ig = Model8.get_loss(params_repr; learning_problem=learning_problem)

@show loss_ig

## ------ intermediate save for very long runs ------ ##

function intermediate_save(intermediate_u, loss_history)
    println("Intermediate save in $(this_run_description) at $(now())")
    @save joinpath(data_path, this_run_description * "_latest.jdl2") intermediate_u loss_history my_config learning_problem
    return nothing
end

## ------ Differential Evolution optimization ------ ##

if my_config["maxiters_bb"] > 0
    bbo_protocol = Model8.BBOProtocol(
        maxiters=my_config["maxiters_bb"],
        intermediate_save=length(my_config["these_on_time_indexes"]) > 1 ? intermediate_save : nothing,
    )
    @time result = Model8.learn(bbo_protocol, learning_problem, params_repr)
    best_p_repr, bb_loss_history = result["parameters"], result["loss_history"]

    Model8.save_learning_result(joinpath(data_path, this_run_description * "_bb.jdl2"), result, my_config, learning_problem)

    # @show learning_problem.model

end

## ------ Nelder-Mead optimization ------ ##

if my_config["maxiters_simplex"] > 0

    # initial guess
    if haskey(my_config, "ig_for_simplex") && !isnothing(my_config["ig_for_simplex"])
        data_file_name =
            my_config["ig_for_simplex"]["run_name"] * "_" * join(my_config["these_on_time_indexes"], "_") * "_bb.jdl2"
        println("Loading initial guess from: ", data_file_name)
        @load joinpath(base_path, "data", my_config["ig_for_simplex"]["data_date"], data_file_name) best_p_repr
        ig_for_simplex = best_p_repr
        @show Model8.get_loss(ig_for_simplex; learning_problem=learning_problem)
    else
        ig_for_simplex = best_p_repr # uses best from differential evolution
    end

    # Load simplex hyperparameters from config
    simplex_params = Model8.load_simplex_hyperparams(my_config)
    simplex_protocol = Model8.SimplexProtocol(
        maxiters=my_config["maxiters_simplex"];
        simplex_params...
    )
    @time result = Model8.learn(simplex_protocol, learning_problem, ig_for_simplex)
    best_p_repr, simplex_loss_history = result["parameters"], result["loss_history"]

    Model8.save_learning_result(joinpath(data_path, this_run_description * "_simplex.jdl2"), result, my_config, learning_problem)
end



best_metrics = Model8.get_metrics(best_p_repr; learning_problem=learning_problem)
println("Best metrics:")
Model8.print_metrics(best_metrics)
println("Best parameters:")
print(best_p_repr)
