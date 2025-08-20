using Model8
using JunTools
using Makie
using CairoMakie
using JLD2

# ================================================================
# CONFIGURATION AND CONSTANTS
# ================================================================

# specify run


# run_name = "linguini1"
# learning_algo = "corduroy"
# model_name = "ModelF8"
# these_on_time_indexes = [1, 2, 3, 4, 5, 6]#
# data_date = "250315"

# Analysis constants
const HARRIS_STDEV_THRESHOLD = 0.0638  # Mean stdev from Harris data (target threshold)
const LOSS_HISTORY_THIN_THRESHOLD = 10000  # Thin loss history if more than this many points
const REFINED_OFF_TIMES_POINTS = 100  # Number of points for refined off_times sweep
const DEFAULT_MOVIE_FRAMES = 100  # Total number of frames for movie
const DEFAULT_FRAMERATE = 30  # Frames per second for movie

# Run specification - can be overridden by command line arguments
# Default values (used when no command line args provided)
run_name = "cocoa2_task3b_modelF8_sweep_job6"
learning_algo = "corduroy"
model_name = "ModelF8"
these_on_time_indexes = [1,2,3,4,5,6]
data_date = "250714"

# run_name = "aioli1_task1b_modelF6"
# learning_algo = "corduroy"
# model_name = "ModelF6"
# these_on_time_indexes = [1,2,3,4,5,6]
# data_date = "250715"

# run_name = "bbq1_task2b_modelF7"
# learning_algo = "corduroy"
# model_name = "ModelF7"
# these_on_time_indexes = [6]
# data_date = "250715"

# Parse command line arguments if provided
# Usage: julia analyze_learning.jl [run_name] [data_date] [learning_algo]
if length(ARGS) >= 1
    run_name = ARGS[1]
    println("Using run_name from command line: $run_name")
end

if length(ARGS) >= 2
    data_date = ARGS[2]
    println("Using data_date from command line: $data_date")
end

if length(ARGS) >= 3
    learning_algo = ARGS[3]
    println("Using learning_algo from command line: $learning_algo")
end

# Set model_name and these_on_time_indexes based on run_name for test runs
if startswith(run_name, "test_")
    if startswith(run_name, "test_classical")
        model_name = "MyModel8"
    elseif startswith(run_name, "test_corduroy")
        model_name = "ModelF8"
    else
        model_name = "MyModel8"  # Default for other test runs
    end
    these_on_time_indexes = [6]
    println("Test run detected - using model_name: $model_name, these_on_time_indexes: $these_on_time_indexes")
end

# ================================================================
# PATH SETUP AND FILE LOADING
# ================================================================

# Setup paths
base_path = JunTools.get_base_path("TCRPulsing")
data_path = joinpath(base_path, "data", data_date)
plot_path = JunTools.get_plot_path("TCRPulsing")

# Load experimental data for error bars
@load joinpath(base_path, "data/Harris_data_CD69.jld2") data
readout_protein = "CD69"
c24stdev_all = data[readout_protein]["c24stdev"]

# Create run description and find file path with pattern matching
this_run_description = "$(run_name)_$(model_name)_$(join(these_on_time_indexes, "_"))"

# Try to find the file using pattern matching to handle both standard and batch script naming
dataset_desc = join(these_on_time_indexes, "_")
file_suffix = "_$(model_name)_$(dataset_desc)_$(learning_algo).jdl2"

# Create regex that matches both patterns:
# 1. run_name_job[INTEGER]_model_name_dataset_desc_algo.jdl2 (from batch scripts)
# 2. run_name_model_name_dataset_desc_algo.jdl2 (standard format)
file_regex = Regex("^" * escape_string(run_name) * "(_job\\d+)?" * escape_string(file_suffix) * "\$")

# Find matching files in the data directory
if !isdir(data_path)
    error("Data directory does not exist: $data_path")
end

all_files = readdir(data_path)
matching_files = filter(f -> occursin(file_regex, f), all_files)

if length(matching_files) == 1
    file_path = joinpath(data_path, matching_files[1])
    println("Found file: $(matching_files[1])")
elseif length(matching_files) == 0
    # Fallback to original naming convention for backward compatibility
    file_path = joinpath(data_path, "$(this_run_description)_$(learning_algo).jdl2")
    if !isfile(file_path)
        error("No files found matching pattern: $(run_name)(_job[INTEGER])?$(file_suffix) or standard format in $data_path")
    end
    println("Using standard naming convention: $(basename(file_path))")
else
    # error("Multiple files found matching pattern: $(run_name)(_job[INTEGER])?$(file_suffix) in $data_path: $matching_files")
    file_path = joinpath(data_path, matching_files[4])
end

# Load using unified interface
result, my_config, learning_problem = Model8.load_learning_result(file_path)

# Subset c24stdev data to match the on_time_indexes used in this run
c24stdev = c24stdev_all[these_on_time_indexes]

best_p_repr = result["parameters"]
loss_history_this_algo = if learning_algo == "corduroy"
    # Special handling for corduroy composite results
    if haskey(result["metadata"], "corduroy_loss_history")
        result["metadata"]["corduroy_loss_history"]
    else
        result["loss_history"]  # Fallback to regular loss history
    end
elseif learning_algo == "pretraining"
    # Handle legacy pretraining files that might not follow standard format
    result["loss_history"]
else
    result["loss_history"]
end

# Handle special cases for legacy compatibility
if learning_algo == "pretraining" && haskey(result["metadata"], "loss_pretraining")
    loss_history_this_algo = result["metadata"]["loss_pretraining"]
end

loss_history = Dict(learning_algo => loss_history_this_algo)

# ================================================================
# METRICS ANALYSIS
# ================================================================

@show typeof(learning_problem.model)

# Check parameter bounds
dataset_desc = length(these_on_time_indexes) == 6 ? "all_datasets" : "dataset_$(join(these_on_time_indexes, "_"))"
bound_violations = Model8.check_parameter_bounds(best_p_repr, learning_problem.model, model_name, dataset_desc)

if isempty(bound_violations)
    println("All parameters within bounds.")
end

metrics = Model8.get_metrics(best_p_repr; learning_problem)
Model8.print_metrics(metrics)

Model8.get_loss(best_p_repr; learning_problem=learning_problem)

println("Time to evaluate get_loss() at best_p_repr:")
@time Model8.get_loss(best_p_repr; learning_problem=learning_problem)
@time Model8.get_loss(best_p_repr; learning_problem=learning_problem)
@time Model8.get_loss(best_p_repr; learning_problem=learning_problem)

# ================================================================
# FREQUENCY RESPONSE ANALYSIS
# ================================================================

p_all_derepresented = Model8.derepresent_all(best_p_repr, learning_problem.model)

# Create frequency response plot using reusable function
@time fig = Model8.plot_frequency_response(
    learning_problem, 
    p_all_derepresented; 
    refined_off_times=true, 
    refined_points=REFINED_OFF_TIMES_POINTS,
    c24stdev=c24stdev
)

display(fig)
save(joinpath(plot_path, this_run_description * "_frequency_response_$(learning_algo).pdf"), fig)

# ================================================================
# LOSS HISTORY ANALYSIS
## ================================================================

if !isnothing(loss_history[learning_algo]) && learning_algo != "corduroy"
    fig_loss_history = Figure()
    ax = Makie.Axis(fig_loss_history[1, 1]; xlabel="Iteration", ylabel="Loss")

    num_datapoints = length(learning_problem.off_times[1]) * length(learning_problem.on_times)
    dirty_rms = sqrt.(loss_history[learning_algo] / num_datapoints) # this is a dirty estimate because the loss function might eventually be affected by other terms.

    if length(loss_history[learning_algo]) < LOSS_HISTORY_THIN_THRESHOLD
        plot!(ax, 1:length(loss_history[learning_algo]), dirty_rms; color=:blue)
    else
        skip = floor(Int, length(loss_history[learning_algo]) / LOSS_HISTORY_THIN_THRESHOLD)
        loss_history_thinned = loss_history[learning_algo][1:skip:end]
        plot!(ax, 1:length(loss_history_thinned), loss_history_thinned; color=:blue)
    end

    Makie.ylims!(ax, 1e-3, 1e3)
    Makie.hlines!(ax, [HARRIS_STDEV_THRESHOLD]; color=:gray, linestyle=:dash)
    ax.yscale = log10

    display(fig_loss_history)
    save(joinpath(plot_path, this_run_description * "_loss_history_$(learning_algo).pdf"), fig_loss_history)

    display(p_all_derepresented.p_classical)
end

if learning_algo == "corduroy"
    fig_loss_history = Figure()
    ax = Makie.Axis(fig_loss_history[1, 1]; xlabel="Iteration", ylabel="Loss")

    round_start = 0
    for round in 1:length(loss_history["corduroy"])
        global round_start
        if loss_history["corduroy"][round][2] == "cmaes"
            println("cmaes")
            losses = loss_history["corduroy"][round][1][2:end]
            plot!(ax, (1:length(losses)) .+ round_start, losses; color=:blue)
        else
            losses = loss_history["corduroy"][round][1]
            plot!(ax, (1:length(losses)) .+ round_start, losses; color=:red)
        end
        round_start += length(losses)
    end

    # Makie.ylims!(ax, 1e-3, 1e3)
    # Makie.hlines!(ax, [HARRIS_STDEV_THRESHOLD]; color=:gray, linestyle=:dash)
    ax.yscale = log10


    display(fig_loss_history)
    save(joinpath(plot_path, this_run_description * "_loss_history_$(learning_algo).pdf"), fig_loss_history)

end

# ================================================================
## REGULATOR FUNCTIONS ANALYSIS
# ================================================================

# Create regulator plot using reusable function from plotting.jl
fig_regulator = Model8.plot_single_model_regulators(
    p_all_derepresented, 
    best_p_repr;
    title="Regulator Functions: $model_name"
)

display(fig_regulator)
save(joinpath(plot_path, this_run_description * "_$(learning_algo)_regulator.pdf"), fig_regulator)

# ================================================================
## MAXIMA-MINIMA AND TIME SERIES 
# ================================================================

# Create movie subdirectory
movie_path = joinpath(plot_path, "movie")
mkpath(movie_path)

# Configure which (on_time, off_time) pairs to save using the new clean interface
# Format: [(on_time_idx, off_time_idx), ...] or `nothing` for all pairs
# on_time_idx: index in learning_problem.on_times (1-based)
# off_time_idx: index in learning_problem.off_times[on_time_idx] (1-based)

# Option 1: Select specific pairs from the refined off_times (memory efficient)
# Note: These indices now refer to the 200-point refined array, not the original sparse data!
save_pairs = [
    (1, 1),     # First on_time, first refined off_time (~min value)
    (1, 50),    # First on_time, 25% through refined range
    (1, 100),   # First on_time, 50% through refined range  
    (1, 200),   # First on_time, last refined off_time (~max value)
    (3, 1),     # Third on_time, first refined off_time
    (6, 200)    # Sixth on_time, last refined off_time
]

# Option 2: Save all refined (on_time, off_time) combinations (comprehensive analysis)
# save_pairs = nothing  # Use this for complete analysis (up to 700 figures)

# Create selection configuration for the new interface
selection = Model8.TimeseriesSelection(
    save_pairs,                    # Which (on_time, off_time) pairs to save
    [:pdf, :png],                  # Save both PDF and PNG formats
    700                            # Limit to 700 figures to prevent excessive output
)

if isnothing(save_pairs)
    total_combinations = sum(length(off_times) for off_times in learning_problem.off_times)
    println("Saving time series for ALL (on_time, off_time) combinations")
    println("  Total possible combinations: $total_combinations (limited to first 300)")
else
    println("Saving time series for selected (on_time_idx, off_time_idx) pairs: $save_pairs")
    for (i, (on_idx, off_idx)) in enumerate(save_pairs)
        if on_idx <= length(learning_problem.on_times) && off_idx <= length(learning_problem.off_times[on_idx])
            println("  Pair $i: on_time=$(learning_problem.on_times[on_idx]) min, off_time=$(learning_problem.off_times[on_idx][off_idx]) min")
        end
    end
end

# Create refined off_times for smooth analysis (200 points for smooth movies)
# This creates 200 linearly spaced points between min/max of each on_time's off_times
refined_off_times = Model8.create_refined_off_times(learning_problem, 200)

# Alternative: Use original sparse off_times for analysis that matches experimental data
# sparse_off_times = learning_problem.off_times  
# Then save_pairs would refer to indices in the sparse experimental arrays

# CLEAN TIMESERIES GENERATION - now with flexible off_times selection!
@time results = Model8.analyze_and_save_timeseries(learning_problem, p_all_derepresented, refined_off_times; selection=selection) do metadata, fig
    # Save PDF for selected pairs
    if metadata.save_pdf
        save(joinpath(plot_path, this_run_description * "_$(learning_algo)_1timeseries_$(metadata.linear_index).pdf"), fig)
    end
    
    # Save PNG for movie frames  
    if metadata.save_png
        save(joinpath(movie_path, "tmp_" * this_run_description * "_$(learning_algo)_1timeseries_$(metadata.linear_index).png"), fig)
    end
end

# Save the extrema plot
display(results.extrema_fig)
save(joinpath(plot_path, this_run_description * "_$(learning_algo)_extrema.pdf"), results.extrema_fig)

println("Processed $(results.num_figures) time series figures")

function print_ffmpeg_command(plot_path, this_run_description, learning_algo, num_frames, framerate=30)
    # Movie subdirectory
    movie_path = joinpath(plot_path, "movie")
    
    # Get the full path to the first frame to check if it exists
    first_frame = joinpath(movie_path, "tmp_$(this_run_description)_$(learning_algo)_1timeseries_1.png")

    if !isfile(first_frame)
        println("Warning: Could not find expected first frame at: $first_frame")
    end

    # Output movie path
    output_movie_path = joinpath(plot_path, "$(this_run_description)_$(learning_algo)_movie.mov")

    # Pattern for input files
    input_pattern = joinpath(movie_path, "tmp_$(this_run_description)_$(learning_algo)_1timeseries_%d.png")

    # Construct the ffmpeg command
    ffmpeg_command = """
    /opt/homebrew/bin/ffmpeg -framerate $framerate -i "$input_pattern" -c:v libx264 -pix_fmt yuv420p -crf 18 "$output_movie_path"
    """

    println("Copy and paste this command into your terminal:")
    println("----------------------------------------------")
    println(ffmpeg_command)
    println("----------------------------------------------")
    println("This will create a movie from $(num_frames) frames at $(framerate) fps")
    println("Output will be saved to: $output_movie_path")

    return nothing
end

# Call the function to print the command
print_ffmpeg_command(plot_path, this_run_description, learning_algo, DEFAULT_MOVIE_FRAMES, DEFAULT_FRAMERATE)

# ================================================================
## TASK 5A: CONTINUOUS PULSES ANALYSIS
# ================================================================
# Additional analysis with continuous_pulses=true as required by Task 5a

println("\n" * "="^70)
println("TASK 5A: CONTINUOUS PULSES ANALYSIS")
println("="^70)

# Create new learning problem with continuous_pulses=true
continuous_learning_problem = LearningProblem(
    learning_problem.on_times,
    learning_problem.off_times, 
    learning_problem.c24_data,
    learning_problem.p_repr_lb,
    learning_problem.p_repr_ub,
    learning_problem.model,
    true,  # continuous_pulses=true
    learning_problem.mask,
    learning_problem.loss_strategy
)

println("Created continuous_pulses learning problem")
println("Original continuous_pulses: $(learning_problem.continuous_pulses)")
println("New continuous_pulses: $(continuous_learning_problem.continuous_pulses)")

# ================================================================
# CONTINUOUS PULSES FREQUENCY RESPONSE
# ================================================================

println("\nGenerating frequency response with continuous_pulses=true...")

# Create frequency response plot using reusable function
@time fig_continuous_freq = Model8.plot_frequency_response(
    continuous_learning_problem, 
    p_all_derepresented; 
    refined_off_times=true, 
    refined_points=REFINED_OFF_TIMES_POINTS,
    title="Task 5a: Frequency Response (continuous_pulses=true)",
    show_data=false  # Hide experimental data for continuous pulses
)

display(fig_continuous_freq)
save(joinpath(plot_path, this_run_description * "_frequency_response_$(learning_algo)_continuous_pulses.pdf"), fig_continuous_freq)
println("Continuous pulses frequency response saved")

# ================================================================
# CONTINUOUS PULSES TIME SERIES
# ================================================================

println("\nGenerating time series with continuous_pulses=true...")

# Create movie subdirectory for continuous pulses
continuous_movie_path = joinpath(plot_path, "movie_continuous")
mkpath(continuous_movie_path)

# Create selection configuration for continuous pulses (reuse same selection as regular analysis)
continuous_selection = Model8.TimeseriesSelection(
    save_pairs,                       # Same selection as regular analysis (nothing = all pairs)
    [:pdf, :png],                     # Save both PDF and PNG formats
    700                               # Limit to 700 figures to prevent excessive output
)

# Create refined off_times for continuous pulses (reuse same refined array)
continuous_refined_off_times = Model8.create_refined_off_times(continuous_learning_problem, 200)

# CLEAN CONTINUOUS TIMESERIES GENERATION - using new interface with refined off_times
@time continuous_results = Model8.analyze_and_save_timeseries(continuous_learning_problem, p_all_derepresented, continuous_refined_off_times; selection=continuous_selection) do metadata, fig
    # Save PDF for selected pairs with continuous_pulses suffix
    if metadata.save_pdf
        save(joinpath(plot_path, this_run_description * "_$(learning_algo)_1timeseries_$(metadata.linear_index)_continuous_pulses.pdf"), fig)
    end
    
    # Save PNG for movie frames with continuous suffix
    if metadata.save_png
        save(joinpath(continuous_movie_path, "tmp_" * this_run_description * "_$(learning_algo)_1timeseries_$(metadata.linear_index)_continuous.png"), fig)
    end
end

# Save the extrema plot for continuous pulses
display(continuous_results.extrema_fig)
save(joinpath(plot_path, this_run_description * "_$(learning_algo)_extrema_continuous_pulses.pdf"), continuous_results.extrema_fig)

println("Processed $(continuous_results.num_figures) continuous pulses time series figures")

# Print ffmpeg command for continuous pulses movie
println("\nFFMPEG command for continuous pulses movie:")
function print_continuous_ffmpeg_command(plot_path, this_run_description, learning_algo, num_frames, framerate=30)
    # Movie subdirectory for continuous pulses
    movie_path = joinpath(plot_path, "movie_continuous")
    
    # Output movie path
    output_movie_path = joinpath(plot_path, "$(this_run_description)_$(learning_algo)_movie_continuous_pulses.mov")
    
    # Pattern for input files
    input_pattern = joinpath(movie_path, "tmp_$(this_run_description)_$(learning_algo)_1timeseries_%d_continuous.png")
    
    # Construct the ffmpeg command
    ffmpeg_command = """
    /opt/homebrew/bin/ffmpeg -framerate $framerate -i "$input_pattern" -c:v libx264 -pix_fmt yuv420p -crf 18 "$output_movie_path"
    """
    
    println("Copy and paste this command into your terminal:")
    println("----------------------------------------------")
    println(ffmpeg_command)
    println("----------------------------------------------")
    println("This will create a continuous pulses movie from $(num_frames) frames at $(framerate) fps")
    println("Output will be saved to: $output_movie_path")
    
    return nothing
end

print_continuous_ffmpeg_command(plot_path, this_run_description, learning_algo, DEFAULT_MOVIE_FRAMES, DEFAULT_FRAMERATE)

println("\n" * "="^70)
println("TASK 5A COMPLETE")
println("="^70)
println("Generated all required visualizations:")
println("✓ FlexiFunctions plots (if applicable)")
println("✓ Learning trajectories (loss history)")
println("✓ Frequency response (original and continuous_pulses=true)")
println("✓ Time series samples (original and continuous_pulses=true)")
println("✓ Extrema plots (original and continuous_pulses=true)")
println("✓ Movie frame PNGs for both pulse modes")
println("\nFiles saved with '_continuous_pulses' suffix for continuous pulse analysis")