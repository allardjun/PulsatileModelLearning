# timeseries_analysis.jl - Modern streaming interface for timeseries analysis

using Base.Iterators

# Configuration structs for flexible timeseries analysis

"""
    TimeseriesSelection

Configuration for selecting which timeseries to process/save.

# Fields
- `save_pairs::Union{Vector{Tuple{Int,Int}}, Nothing}`: (on_time_idx, off_time_idx) pairs to save, or `nothing` for all pairs
- `save_formats::Vector{Symbol}`: Formats to save (:pdf, :png, etc.)
- `max_count::Union{Int,Nothing}`: Maximum number of figures to process
"""
struct TimeseriesSelection
    save_pairs::Union{Vector{Tuple{Int,Int}}, Nothing}
    save_formats::Vector{Symbol}
    max_count::Union{Int,Nothing}
end

function default_timeseries_selection()
    return TimeseriesSelection(
        [(1, 1), (1, 7), (3, 1), (6, 14)],
        [:pdf],
        nothing
    )
end

"""
    all_timeseries_selection(; save_formats=[:pdf], max_count=300)

Create a TimeseriesSelection that processes ALL (on_time, off_time) combinations.
Useful for comprehensive analysis or when you want to save all figures.
Default limit of 300 figures prevents excessive output.

# Example
```julia
selection = PulsatileModelLearning.all_timeseries_selection(save_formats=[:pdf, :png])
selection_unlimited = PulsatileModelLearning.all_timeseries_selection(max_count=nothing)  # No limit
```
"""
function all_timeseries_selection(; save_formats=[:pdf], max_count=300)
    return TimeseriesSelection(nothing, save_formats, max_count)
end

"""
    TimeseriesMetadata

Metadata for a single timeseries figure.

# Fields
- `on_time_idx::Int`: Index in on_times array
- `off_time_idx::Int`: Index in off_times[on_time_idx] array
- `linear_index::Int`: Linear index across all combinations
- `on_time::Real`: Actual on_time value (minutes)
- `off_time::Real`: Actual off_time value (minutes)  
- `save_pdf::Bool`: Whether this figure should be saved as PDF
- `save_png::Bool`: Whether this figure should be saved as PNG
"""
struct TimeseriesMetadata
    on_time_idx::Int
    off_time_idx::Int
    linear_index::Int
    on_time::Real
    off_time::Real
    save_pdf::Bool
    save_png::Bool
end

"""
    GlobalExtrema

Container for global extrema data across all timeseries.

# Fields
- `maxima::Array{Real,3}`: Maximum values [on_time_idx, off_time_idx, variable_idx]
- `minima::Array{Real,3}`: Minimum values [on_time_idx, off_time_idx, variable_idx]
- `max_values::Vector{Real}`: Global maximum for each variable (for y-axis scaling)
"""
struct GlobalExtrema
    maxima::Array{Real,3}
    minima::Array{Real,3}
    max_values::Vector{Real}
end

"""
    compute_global_extrema(on_times, off_times, p_derepresented, model; continuous_pulses=false)

Efficiently compute global extrema across all (on_time, off_time) combinations.
This is used for consistent y-axis scaling across all timeseries figures.

# Returns
- `GlobalExtrema`: Container with maxima, minima, and global max values
"""
function compute_global_extrema(on_times, off_times, p_derepresented, model; continuous_pulses=false)
    maxima = zeros(Real, length(on_times), length(off_times[1]), 4)
    minima = zeros(Real, length(on_times), length(off_times[1]), 4)
    
    for (idx_on_time, on_time) in enumerate(on_times)
        for (idx_off_time, off_time) in enumerate(off_times[idx_on_time])
            i_func = t -> i_pulses(t, on_time, off_time; continuous_pulses=continuous_pulses)
            rhs = PulsatileModelLearning.make_rhs(i_func, model)
            prob = ODEProblem(rhs, model.u0, [0.0, 24 * 60.0])
            ode_solution = solve(prob, Vern9(); p=p_derepresented, abstol=1e-8, reltol=1e-8)
            
            u = Array(ode_solution)
            maxima[idx_on_time, idx_off_time, :] = [maximum(u[i, :]) for i in 1:4]
            minima[idx_on_time, idx_off_time, :] = [minimum(u[i, :]) for i in 1:4]
        end
    end
    
    # Compute global maxima for y-axis scaling
    max_values = [maximum(maxima[:, :, i]) for i in 1:4]
    
    return GlobalExtrema(maxima, minima, max_values)
end

"""
    TimeseriesGenerator

Iterator that generates timeseries figures on-demand with global y-axis scaling.
Memory-efficient alternative to storing all figures in memory.

# Fields
- `on_times`: Array of on_time values
- `off_times`: Array of off_time arrays (one per on_time)
- `p_derepresented`: Derepresented parameters  
- `model`: Model instance
- `global_extrema::GlobalExtrema`: Pre-computed extrema for y-axis scaling
- `selection::TimeseriesSelection`: Selection configuration
- `continuous_pulses::Bool`: Whether to use continuous pulses
"""
struct TimeseriesGenerator
    on_times
    off_times
    p_derepresented
    model
    global_extrema::GlobalExtrema
    selection::TimeseriesSelection
    continuous_pulses::Bool
end

function Base.iterate(gen::TimeseriesGenerator, state=(1, 1, 1))
    on_idx, off_idx, linear_idx = state
    
    # Check if we've processed all combinations
    if on_idx > length(gen.on_times)
        return nothing
    end
    
    # Check if we've processed all off_times for current on_time
    if off_idx > length(gen.off_times[on_idx])
        next_state = (on_idx + 1, 1, linear_idx)
        return iterate(gen, next_state)
    end
    
    # Check if we should process this combination based on selection criteria
    current_pair = (on_idx, off_idx)
    should_process = if isnothing(gen.selection.save_pairs)
        true  # Process all pairs if save_pairs is nothing
    else
        current_pair in gen.selection.save_pairs
    end
    
    if !should_process
        # Skip to next combination
        next_state = (on_idx, off_idx + 1, linear_idx + 1)
        return iterate(gen, next_state)
    end
    
    # Check max_count limit
    if !isnothing(gen.selection.max_count) && linear_idx > gen.selection.max_count
        return nothing
    end
    
    # Generate the figure for this combination
    on_time = gen.on_times[on_idx]
    off_time = gen.off_times[on_idx][off_idx]
    
    # Solve ODE
    i_func = t -> i_pulses(t, on_time, off_time; continuous_pulses=gen.continuous_pulses)
    rhs = PulsatileModelLearning.make_rhs(i_func, gen.model)
    prob = ODEProblem(rhs, gen.model.u0, [0.0, 24 * 60.0])
    ode_solution = solve(prob, Vern9(); p=gen.p_derepresented, abstol=1e-8, reltol=1e-8)
    
    # Create figure with global y-axis scaling
    t = ode_solution.t
    u = Array(ode_solution)
    fig = create_timeseries_figure_with_global_scaling(t, u, i_func, gen.p_derepresented, 
                                                       gen.global_extrema.max_values;
                                                       on_time=on_time, off_time=off_time)
    
    # Create metadata
    metadata = TimeseriesMetadata(
        on_idx, off_idx, linear_idx,
        on_time, off_time,
        :pdf in gen.selection.save_formats,
        :png in gen.selection.save_formats
    )
    
    # Prepare next state
    next_state = (on_idx, off_idx + 1, linear_idx + 1)
    
    return ((metadata, fig), next_state)
end

Base.IteratorSize(::Type{TimeseriesGenerator}) = Base.SizeUnknown()

"""
    analyze_timeseries_streaming(on_times, off_times, p_derepresented, model;
                                selection=default_timeseries_selection(),
                                continuous_pulses=false)

Create a memory-efficient streaming generator for timeseries analysis.
Figures are generated on-demand with consistent global y-axis scaling.

# Arguments
- `on_times`: Array of on_time values
- `off_times`: Array of off_time arrays  
- `p_derepresented`: Derepresented parameters
- `model`: Model instance
- `selection`: Selection configuration (which figures to generate)
- `continuous_pulses`: Whether to use continuous pulses

# Returns
- `TimeseriesGenerator`: Iterator yielding (metadata, figure) pairs

# Example
```julia
selection = TimeseriesSelection([(1,1), (1,7), (3,1), (6,14)], [:pdf, :png], "movie", nothing)

for (metadata, fig) in analyze_timeseries_streaming(on_times, off_times, p_all, model; selection)
    if metadata.save_pdf
        save("timeseries_\$(metadata.linear_index).pdf", fig)
    end
    if metadata.save_png
        save("movie/frame_\$(metadata.linear_index).png", fig)
    end
end
```
"""
function analyze_timeseries_streaming(on_times, off_times, p_derepresented, model;
                                     selection=default_timeseries_selection(),
                                     continuous_pulses=false)
    
    # Pre-compute global extrema for consistent y-axis scaling
    global_extrema = compute_global_extrema(on_times, off_times, p_derepresented, model; continuous_pulses)
    
    return TimeseriesGenerator(on_times, off_times, p_derepresented, model,
                              global_extrema, selection, continuous_pulses)
end

"""
    create_timeseries_figure_with_global_scaling(t, u, i_func, p_derepresented, max_values; title=nothing, on_time=nothing, off_time=nothing)

Create a timeseries figure with the standard 5-panel layout and global y-axis scaling.
Fixed layout: i(t), a, m, w, c in vertical panels (exactly 5 panels).

# Arguments
- `t`: Time vector from ODE solution
- `u`: State variables array (4×N matrix: a, m, w, c)
- `i_func`: Input function
- `p_derepresented`: Derepresented parameters (for threshold lines)
- `max_values::Vector{Real}`: Global maximum values for y-axis scaling
- `title`: Optional title for the entire figure
- `on_time`: Optional on_time value for automatic title generation (3 digits, no decimals)
- `off_time`: Optional off_time value for automatic title generation (3 digits, no decimals)

# Returns
- `fig`: Makie Figure with exactly 5 panels and global scaling

# Layout
1. i(t) - Input function (black)
2. a - First state variable (red)
3. m - Second state variable (green)
4. w - Third state variable (blue)
5. c - Fourth state variable (orange)

Note: This function creates exactly 5 panels. For layouts with optional w-inhib panels, 
use the legacy create_timeseries_figure() function.
"""
function create_timeseries_figure_with_global_scaling(t, u, i_func, p_derepresented, max_values; title=nothing, on_time=nothing, off_time=nothing)
    fig = Figure(; size=(400, 1000))
    
    # Generate title if on_time and off_time are provided
    if !isnothing(on_time) && !isnothing(off_time)
        title = "Ton=$(round(Int, on_time)) Toff=$(round(Int, off_time))"
    end
    
    # Add title if provided
    if !isnothing(title)
        fig[0, :] = Label(fig, title, fontsize=12, tellwidth=false)
    end
    
    # Panel 1: Input function i(t) - BLACK
    ax_input = Makie.Axis(fig[1, 1]; xlabel="Time (min)")
    i_values = [i_func(t_val) for t_val in t]
    lines!(ax_input, t, i_values; color=:black, linewidth=1.5)
    ax_input.ylabel = "i(t)"
    
    # Panel 2: Variable a - RED
    ax_a = Makie.Axis(fig[2, 1]; xlabel="Time (min)")
    lines!(ax_a, t, u[1, :]; color=:red, linewidth=1.5)
    ax_a.ylabel = "a"
    Makie.ylims!(ax_a, 0.0, 1.05 * max_values[1])
    # Add threshold line at 1/beta_mw if parameter exists
    if haskey(p_derepresented.p_classical, :beta_mw)
        beta_mw = p_derepresented.p_classical.beta_mw
        hlines!(ax_a, [1 / beta_mw]; color=:gray, linestyle=:dash, linewidth=1)
    end
    
    # Panel 3: Variable m - GREEN
    ax_m = Makie.Axis(fig[3, 1]; xlabel="Time (min)")
    lines!(ax_m, t, u[2, :]; color=:green, linewidth=1.5)
    ax_m.ylabel = "m"
    Makie.ylims!(ax_m, 0.0, 1.05 * max_values[2])
    
    # Panel 4: Variable w - BLUE
    ax_w = Makie.Axis(fig[4, 1]; xlabel="Time (min)")
    lines!(ax_w, t, u[3, :]; color=:blue, linewidth=1.5)
    ax_w.ylabel = "w"
    Makie.ylims!(ax_w, 0.0, max(1.05 * max_values[3], 0.001))
    # Add threshold line at 1/beta_ma if parameter exists
    if haskey(p_derepresented.p_classical, :beta_ma)
        beta_ma = p_derepresented.p_classical.beta_ma
        hlines!(ax_w, [1 / beta_ma]; color=:gray, linestyle=:dash, linewidth=1)
    end
    
    # Panel 5: Variable c - ORANGE
    ax_c = Makie.Axis(fig[5, 1]; xlabel="Time (min)")
    lines!(ax_c, t, u[4, :]; color=:orange, linewidth=1.5)
    # Add large orange circle at final time point
    scatter!(ax_c, [t[end]], [u[4, end]]; color=:orange, markersize=12)
    ax_c.ylabel = "c"
    Makie.ylims!(ax_c, 0.0, 1.05 * max_values[4])
    
    return fig
end

"""
    AnalysisResults

Results container for high-level timeseries analysis functions.

# Fields
- `extrema_fig`: Figure showing maxima/minima plots
- `num_figures::Int`: Number of timeseries figures processed
- `global_extrema::GlobalExtrema`: Extrema data used for analysis
"""
struct AnalysisResults
    extrema_fig
    num_figures::Int
    global_extrema::GlobalExtrema
end

"""
    analyze_and_save_timeseries(save_callback, learning_problem, p_all_derepresented, off_times_for_analysis;
                               selection=default_timeseries_selection())

High-level function for timeseries analysis with automatic figure saving.
Combines extrema computation, streaming generation, and custom save logic.

# Arguments
- `save_callback`: Function called as save_callback(metadata, fig) for each figure
- `learning_problem`: LearningProblem containing on_times, model, continuous_pulses, etc.
- `p_all_derepresented`: Derepresented parameters
- `off_times_for_analysis`: Array of off_time arrays to use for analysis (can be sparse or refined)
- `selection`: Selection configuration (save_pairs indices refer to off_times_for_analysis)

# Returns
- `AnalysisResults`: Container with extrema figure and processing statistics

# Example
```julia
# Use original sparse off_times
results = analyze_and_save_timeseries(learning_problem, p_all, learning_problem.off_times) do metadata, fig
    save("timeseries_\$(metadata.linear_index).pdf", fig)
end

# Use refined off_times for smooth analysis
refined_off_times = create_refined_off_times(learning_problem, 200)
results = analyze_and_save_timeseries(learning_problem, p_all, refined_off_times) do metadata, fig
    save("smooth_timeseries_\$(metadata.linear_index).pdf", fig) 
end
```
"""
function analyze_and_save_timeseries(save_callback, learning_problem, p_all_derepresented, off_times_for_analysis;
                                    selection=default_timeseries_selection())
    
    # Create streaming generator using provided off_times
    timeseries_gen = analyze_timeseries_streaming(
        learning_problem.on_times,
        off_times_for_analysis,
        p_all_derepresented,
        learning_problem.model;
        selection=selection,
        continuous_pulses=learning_problem.continuous_pulses
    )
    
    # Process figures one at a time
    num_figures = 0
    for (metadata, fig) in timeseries_gen
        save_callback(metadata, fig)
        num_figures += 1
    end
    
    # Create extrema figure using the analysis off_times
    extrema_fig = create_extrema_figure(timeseries_gen.global_extrema, off_times_for_analysis)
    
    return AnalysisResults(extrema_fig, num_figures, timeseries_gen.global_extrema)
end

"""
    create_refined_off_times(learning_problem, num_points)

Create refined off_times arrays with linearly spaced points between min and max of each on_time.
Useful for smooth timeseries analysis and movie generation.

# Arguments
- `learning_problem`: LearningProblem containing original sparse off_times
- `num_points`: Number of linearly spaced points to create for each on_time

# Returns
- Array of refined off_time arrays, one per on_time

# Example
```julia
# Create 200 refined points for smooth movie frames
refined_off_times = create_refined_off_times(learning_problem, 200)

# Use specific indices from refined array for PDF plots
selection = TimeseriesSelection([(1, 50), (3, 150)], [:pdf], 10)
```
"""
function create_refined_off_times(learning_problem, num_points)
    refined_off_times = []
    for i in axes(learning_problem.off_times, 1)
        push!(
            refined_off_times,
            collect(range(
                minimum(learning_problem.off_times[i][:]), 
                maximum(learning_problem.off_times[i][:]), 
                num_points
            ))
        )
    end
    return refined_off_times
end

"""
    create_extrema_figure(global_extrema, off_times_for_plot)

Create a figure showing maxima and minima for each variable.

# Arguments
- `global_extrema::GlobalExtrema`: Extrema data
- `off_times_for_plot`: Off times used for x-axis

# Returns
- `fig`: Makie Figure with extrema plots
"""
function create_extrema_figure(global_extrema, off_times_for_plot)
    fig = Figure()
    
    variable_names = ["a", "m", "w", "c"]
    for (i, var_name) in enumerate(variable_names)
        ax = Makie.Axis(fig[i, 1]; 
                       xlabel=i == length(variable_names) ? "Interval between pulses Toff (min)" : "",
                       ylabel=var_name)
        
        if !isempty(off_times_for_plot) && size(global_extrema.maxima, 1) >= 1
            scatter!(ax, off_times_for_plot[1], global_extrema.maxima[1, :, i]; label="max", markersize=8)
            scatter!(ax, off_times_for_plot[1], global_extrema.minima[1, :, i]; label="min", markersize=8)
            if i == 1  # Add legend only to first subplot
                axislegend(ax; position=:rt)
            end
        end
    end
    
    return fig
end