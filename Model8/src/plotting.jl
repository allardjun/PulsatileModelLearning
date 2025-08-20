# plotting.jl - Reusable plotting functions for TCR modeling analysis

using Makie

"""
    plot_frequency_response(learning_problem, p_all_derepresented; 
                           refined_off_times=true, refined_points=100, 
                           title=nothing, figure_size=(800, 600), show_data=true, c24stdev=nothing)

Create a frequency response plot showing model fit vs experimental data.

# Arguments
- `learning_problem`: LearningProblem containing experimental data and model setup
- `p_all_derepresented`: Derepresented parameters for the model
- `refined_off_times`: Whether to use refined off_times for smooth curves (default: true)
- `refined_points`: Number of points for refined off_times sweep (default: 100)
- `title`: Optional title for the plot (default: auto-generated)
- `figure_size`: Figure size as (width, height) tuple (default: (800, 600))
- `show_data`: Whether to show experimental data points (default: true)
- `c24stdev`: Optional standard deviation data for error bars (default: nothing)

# Returns
- `fig`: Makie Figure object ready for display or saving

# Example
```julia
p_all = Model8.derepresent_all(best_p_repr, learning_problem.model)
fig = Model8.plot_frequency_response(learning_problem, p_all)
display(fig)
save("frequency_response.pdf", fig)
```
"""
function plot_frequency_response(learning_problem, p_all_derepresented; 
                                refined_off_times=true, 
                                refined_points=100,
                                title=nothing,
                                figure_size=(800, 600),
                                show_data=true,
                                c24stdev=nothing)
    
    # Prepare off_times for plotting (refined or original)
    if refined_off_times
        off_times_for_plot = []
        for i in axes(learning_problem.off_times, 1)
            push!(
                off_times_for_plot,
                collect(range(
                    minimum(learning_problem.off_times[i][:]), 
                    maximum(learning_problem.off_times[i][:]), 
                    refined_points
                )),
            )
        end
    else
        off_times_for_plot = learning_problem.off_times
    end
    
    # Compute frequency response
    c24_normalized = get_freq_response(
        learning_problem.on_times, 
        off_times_for_plot, 
        p_all_derepresented, 
        learning_problem.model; 
        make_timeseries=false,
        continuous_pulses=learning_problem.continuous_pulses,
    )
    
    # Create figure
    fig = Figure(size=figure_size)
    
    # Handle title properly (avoid passing nothing to Makie)
    axis_kwargs = (
        xlabel="Interval between pulses Toff (min)",
        ylabel="Final concentration c(t=24hrs), normalized to tophat",
    )
    if !isnothing(title)
        axis_kwargs = merge(axis_kwargs, (title=title,))
    end
    
    ax = Makie.Axis(fig[1, 1]; axis_kwargs...)
    
    # Plot data and model fits
    colors = cgrad(:auto, length(learning_problem.on_times); categorical=true)
    
    for idx_on_time in eachindex(learning_problem.on_times)
        # Experimental data points (conditional)
        if show_data
            scatter!(
                ax,
                learning_problem.off_times[idx_on_time][:],
                learning_problem.c24_data[idx_on_time, :];
                color=colors[idx_on_time],
                markersize=10,
                label="Data: Ton = " * string(learning_problem.on_times[idx_on_time]) * " min",
            )

            # Add error bars if c24stdev data is provided
            if !isnothing(c24stdev)
                errorbars!(
                    ax, 
                    learning_problem.off_times[idx_on_time][:], 
                    learning_problem.c24_data[idx_on_time, :], 
                    c24stdev[idx_on_time][:]./sqrt(3),
                    color = colors[idx_on_time],
                    alpha = 0.75
                )
            end

        end
        
        # Model fit curves
        lines!(
            ax, 
            off_times_for_plot[idx_on_time][:], 
            c24_normalized[idx_on_time, :]; 
            color=colors[idx_on_time], 
            alpha=0.75,
            label=show_data ? "Model fit" : "Ton = " * string(learning_problem.on_times[idx_on_time]) * " min"
        )
    end
    
    axislegend(ax; position=:rb)
    
    return fig
end

"""
    plot_frequency_response_comparison(learning_problem_1, p_all_1, label_1,
                                     learning_problem_2, p_all_2, label_2;
                                     refined_off_times=true, refined_points=100,
                                     title=nothing, figure_size=(1000, 600), c24stdev=nothing)

Create a frequency response comparison plot between two models.

# Arguments
- `learning_problem_1`: First LearningProblem
- `p_all_1`: First derepresented parameters
- `label_1`: Label for first model (e.g., "Classical")
- `learning_problem_2`: Second LearningProblem  
- `p_all_2`: Second derepresented parameters
- `label_2`: Label for second model (e.g., "Flexi")
- `refined_off_times`: Whether to use refined off_times for smooth curves (default: true)
- `refined_points`: Number of points for refined off_times sweep (default: 100)
- `title`: Optional title for the plot (default: auto-generated)
- `figure_size`: Figure size as (width, height) tuple (default: (1000, 600))
- `c24stdev`: Optional standard deviation data for error bars (default: nothing)

# Returns
- `fig`: Makie Figure object ready for display or saving

# Example
```julia
fig = Model8.plot_frequency_response_comparison(
    classical_learning_problem, classical_p_all, "Classical",
    flexi_learning_problem, flexi_p_all, "Flexi"
)
```
"""
function plot_frequency_response_comparison(learning_problem_1, p_all_1, label_1,
                                          learning_problem_2, p_all_2, label_2;
                                          refined_off_times=true,
                                          refined_points=100,
                                          title=nothing,
                                          figure_size=(1000, 600),
                                          c24stdev=nothing)
    
    # Use first learning problem for experimental data (should be same for both)
    learning_problem = learning_problem_1
    
    # Prepare off_times for plotting
    if refined_off_times
        off_times_for_plot = []
        for i in axes(learning_problem.off_times, 1)
            push!(
                off_times_for_plot,
                collect(range(
                    minimum(learning_problem.off_times[i][:]), 
                    maximum(learning_problem.off_times[i][:]), 
                    refined_points
                )),
            )
        end
    else
        off_times_for_plot = learning_problem.off_times
    end
    
    # Compute frequency responses for both models
    c24_1 = get_freq_response(
        learning_problem_1.on_times, off_times_for_plot, p_all_1, 
        learning_problem_1.model; make_timeseries=false,
        continuous_pulses=learning_problem_1.continuous_pulses
    )
    
    c24_2 = get_freq_response(
        learning_problem_2.on_times, off_times_for_plot, p_all_2,
        learning_problem_2.model; make_timeseries=false,
        continuous_pulses=learning_problem_2.continuous_pulses
    )
    
    # Create figure
    fig = Figure(size=figure_size)
    
    # Handle title properly (avoid passing nothing to Makie)
    axis_kwargs = (
        xlabel="Interval between pulses Toff (min)",
        ylabel="Final concentration c(t=24hrs), normalized to tophat",
    )
    if !isnothing(title)
        axis_kwargs = merge(axis_kwargs, (title=title,))
    end
    
    ax = Makie.Axis(fig[1, 1]; axis_kwargs...)
    
    # Plot data and both model fits
    colors = cgrad(:auto, length(learning_problem.on_times); categorical=true)
    
    for idx_on_time in eachindex(learning_problem.on_times)
        # Experimental data points
        scatter!(
            ax,
            learning_problem.off_times[idx_on_time][:],
            learning_problem.c24_data[idx_on_time, :];
            color=colors[idx_on_time],
            markersize=10,
            label="Data: Ton = $(learning_problem.on_times[idx_on_time]) min"
        )
        
        # Add error bars if c24stdev data is provided
        if !isnothing(c24stdev)
            errorbars!(
                ax, 
                learning_problem.off_times[idx_on_time][:], 
                learning_problem.c24_data[idx_on_time, :], 
                c24stdev[idx_on_time][:]./sqrt(3),
                color = colors[idx_on_time],
                alpha = 0.75
            )
        end
        
        # First model fit
        lines!(
            ax, 
            off_times_for_plot[idx_on_time][:], 
            c24_1[idx_on_time, :]; 
            color=colors[idx_on_time], 
            linestyle=:solid,
            linewidth=2,
            alpha=0.8,
            label="$label_1 fit"
        )
        
        # Second model fit
        lines!(
            ax,
            off_times_for_plot[idx_on_time][:],
            c24_2[idx_on_time, :];
            color=colors[idx_on_time],
            linestyle=:dash,
            linewidth=2,
            alpha=0.8,
            label="$label_2 fit"
        )
    end
    
    axislegend(ax; position=:rb)
    
    return fig
end

# Time series plotting functions using callback approach

"""
    process_and_save_timeseries_figures(save_callback, analysis, p_all_derepresented; 
                                       display_final=true, add_threshold_lines=true, 
                                       return_final_fig=true)

LEGACY FUNCTION: Memory-efficient processing of time series figures using callback pattern.
Maintains backward compatibility with existing code that uses analysis["figs"].

# Arguments
- `save_callback`: Function called as save_callback(on_time_idx, fig) for each processed figure
- `analysis`: Dictionary containing analysis results with "figs", "maxima", "minima" keys
- `p_all_derepresented`: Derepresented parameters for threshold line plotting
- `display_final`: Whether to display the final (last) figure (default: true)
- `add_threshold_lines`: Whether to add horizontal threshold lines (default: true)
- `return_final_fig`: Whether to return the final figure for further use (default: true)

# Returns
- `NamedTuple` with:
  - `extrema_fig`: Figure showing maxima/minima plots
  - `final_timeseries_fig`: The final time series figure (if return_final_fig=true, nothing otherwise)
  - `maxima`: Maxima data from analysis
  - `minima`: Minima data from analysis
  - `num_figures`: Number of time series figures processed

# Example
```julia
_, analysis = Model8.get_freq_response(...; make_timeseries=true, get_extrema=true)
p_all = Model8.derepresent_all(best_p_repr, model)

results = Model8.process_and_save_timeseries_figures(analysis, p_all) do i, fig
    save("timeseries_\$i.pdf", fig)
    save("movie/frame_\$i.png", fig)  # Can save to multiple locations
end

# Save the extrema plot
save("extrema.pdf", results.extrema_fig)

# The final timeseries is available for display/saving
if !isnothing(results.final_timeseries_fig)
    display(results.final_timeseries_fig)
end
```
"""
function process_and_save_timeseries_figures(save_callback, analysis, p_all_derepresented; 
                                            display_final=true, 
                                            add_threshold_lines=true,
                                            return_final_fig=true)
    
    # Extract data from analysis
    figs = analysis["figs"]
    maxima = analysis["maxima"]
    minima = analysis["minima"]
    off_times_for_plot = haskey(analysis, "off_times_for_plot") ? analysis["off_times_for_plot"] : nothing
    
    # Create extrema figure
    fig_maxima = Figure()
    
    # Plot maxima/minima for each variable (a, m, w, c)
    variable_names = ["a", "m", "w", "c"]
    for (i, var_name) in enumerate(variable_names)
        ax = Makie.Axis(fig_maxima[i, 1]; 
                       xlabel=i == length(variable_names) ? "Interval between pulses Toff (min)" : "",
                       ylabel=var_name)
        
        if !isnothing(off_times_for_plot) && size(maxima, 1) >= 1
            scatter!(ax, off_times_for_plot[1], maxima[1, :, i]; label="max", markersize=8)
            scatter!(ax, off_times_for_plot[1], minima[1, :, i]; label="min", markersize=8)
            if i == 1  # Add legend only to first subplot
                axislegend(ax; position=:rt)
            end
        end
    end
    
    # Calculate y-axis limits from maxima
    max_values = [maximum(maxima[1, :, i]) for i in 1:size(maxima, 3)]
    
    # Track the final figure for return (if requested)
    final_fig = nothing
    
    # Process each time series figure one at a time (memory efficient)
    for (on_time_idx, fig) in enumerate(figs)
        # Process the figure in place
        processed_fig = process_single_timeseries_figure!(
            fig, max_values, p_all_derepresented, 
            on_time_idx, length(figs), 
            display_final, add_threshold_lines
        )
        
        # Store final figure if requested
        if return_final_fig && on_time_idx == length(figs)
            final_fig = processed_fig
        end
        
        # Call user's save callback
        save_callback(on_time_idx, processed_fig)
        
        # Note: figure can be garbage collected after this iteration
        # (unless it's the final one we're keeping)
    end
    
    return (
        extrema_fig = fig_maxima,
        final_timeseries_fig = final_fig,
        maxima = maxima,
        minima = minima,
        num_figures = length(figs)
    )
end

"""
    process_single_timeseries_figure!(fig, max_values, p_all_derepresented, 
                                     on_time_idx, total_figs, display_final, add_threshold_lines)

Process a single time series figure in place, adding labels, limits, and threshold lines.
Assumes 5+ panel layout: i(t), a, m, w, c, [w-inhib]
Helper function for process_and_save_timeseries_figures.
"""
function process_single_timeseries_figure!(fig, max_values, p_all_derepresented, 
                                          on_time_idx, total_figs, 
                                          display_final, add_threshold_lines)
    
    # Access the axes (legacy 5+ panel layout: i(t), a, m, w, c, [w-inhib])
    ax_input = fig.content[1, 1]  # i(t) - input function
    ax1 = fig.content[2, 1]       # a
    ax2 = fig.content[3, 1]       # m  
    ax3 = fig.content[4, 1]       # w
    ax4 = fig.content[5, 1]       # c
    
    # Set axis labels
    ax_input.ylabel = "i(t)"
    ax1.ylabel = "a"
    ax2.ylabel = "m"
    ax3.ylabel = "w"
    ax4.ylabel = "c"
    
    # Set y-axis limits based on maxima
    Makie.ylims!(ax1, 0.0, 1.05 * max_values[1])
    Makie.ylims!(ax2, 0.0, 1.05 * max_values[2])
    Makie.ylims!(ax3, 0.0, max(1.05 * max_values[3], 0.001))
    Makie.ylims!(ax4, 0.0, 1.05 * max_values[4])
    
    # Add threshold lines if requested
    if add_threshold_lines
        # Add horizontal line at 1/beta_mw for 'a' variable
        if haskey(p_all_derepresented.p_classical, :beta_mw)
            beta_mw = p_all_derepresented.p_classical.beta_mw
            hlines!(ax1, [1 / beta_mw]; color=:gray, linestyle=:dash, linewidth=1)
        end
        
        # Add horizontal line at 1/beta_ma for 'w' variable  
        if haskey(p_all_derepresented.p_classical, :beta_ma)
            beta_ma = p_all_derepresented.p_classical.beta_ma
            hlines!(ax3, [1 / beta_ma]; color=:gray, linestyle=:dash, linewidth=1)
        end
    end
    
    # Display the final figure if requested
    if display_final && on_time_idx == total_figs
        display(fig)
    end
    
    return fig
end

"""
    create_timeseries_figure(t, u, i_func, p_derepresented)

Create a time series figure showing input function and ODE state variables over time.

# Arguments
- `t`: Time vector from ODE solution
- `u`: State variables array from ODE solution (4×N matrix: a, m, w, c)
- `i_func`: Input function to evaluate over time points
- `p_derepresented`: Derepresented parameters for optional w-inhib calculation

# Returns
- `fig`: Makie Figure with 5+ subplots showing i(t), a, m, w, c, and optionally w-inhib

# Layout
1. i(t) - Input function
2. a - First state variable
3. m - Second state variable
4. w - Third state variable
5. c - Fourth state variable
6. w-inhib - Optional inhibition function (if beta_mw and nwm parameters exist)
"""
function create_timeseries_figure(t, u, i_func, p_derepresented)
    fig = Figure(; size=(400, 1000))

    # Input function axis (first axis)
    ax = Makie.Axis(fig[1, 1]; xlabel="Time (min)")
    i_values = [i_func(t_val) for t_val in t]
    lines!(ax, t, i_values)
    ax.ylabel = "i(t)"
    
    # 4 separate axes for ODE outputs
    ax = Makie.Axis(fig[2, 1]; xlabel="Time (min)")
    lines!(ax, t, u[1, :])
    ax.ylabel = "a"
    ax = Makie.Axis(fig[3, 1]; xlabel="Time (min)")
    lines!(ax, t, u[2, :])
    ax.ylabel = "m"
    ax = Makie.Axis(fig[4, 1]; xlabel="Time (min)")
    lines!(ax, t, u[3, :])
    ax.ylabel = "w"
    ax = Makie.Axis(fig[5, 1]; xlabel="Time (min)")
    lines!(ax, t, u[4, :])
    ax.ylabel = "c"

    # Optional w-inhib axis if parameters exist
    if haskey(p_derepresented.p_classical, :beta_mw) && haskey(p_derepresented.p_classical, :nwm)
        beta_mw = p_derepresented.p_classical.beta_mw
        nwm = p_derepresented.p_classical.nwm
        w = u[3, :]
        w_inhib = 1 ./ (1 .+ (abs.(beta_mw * w)) .^ nwm)
        ax = Makie.Axis(fig[6, 1]; xlabel="Time (min)")
        lines!(ax, t, w_inhib)
        ax.ylabel = "w-inhib"
    end

    return fig
end

"""
    plot_regulator_comparison(classical_p_all, classical_label, flexi_p_repr, flexi_label;
                             title=nothing, figure_size=(1000, 600))

Create a regulator comparison plot between classical and flexi models.

# Arguments
- `classical_p_all`: Classical derepresented parameters
- `classical_label`: Label for classical model (e.g., "Model 6")
- `flexi_p_repr`: Flexi represented parameters (for accessing flex params)
- `flexi_label`: Label for flexi model (e.g., "ModelF6")
- `title`: Optional title for the plot (default: auto-generated)
- `figure_size`: Figure size as (width, height) tuple (default: (1000, 600))

# Returns
- `fig`: Makie Figure object ready for display or saving

# Example
```julia
fig = Model8.plot_regulator_comparison(
    classical_p_all, "Model 6",
    flexi_p_repr, "ModelF6"
)
```
"""
function plot_regulator_comparison(classical_p_all, classical_label, flexi_p_repr, flexi_label;
                                  title=nothing,
                                  figure_size=(1000, 600))
    
    # Create figure
    fig = Figure(size=figure_size)
    
    # Handle title properly (avoid passing nothing to Makie)
    axis_kwargs = (
        xlabel="Input variable",
        ylabel="Regulator function",
    )
    if !isnothing(title)
        axis_kwargs = merge(axis_kwargs, (title=title,))
    end
    
    # Check if we have classical regulators
    has_regulator1_classical = haskey(classical_p_all.p_classical, :beta_mw)
    has_regulator2_classical = haskey(classical_p_all.p_classical, :beta_ma)
    
    # Check if we have flexi regulators (model-specific mapping)
    # Model-specific flex parameter mapping:
    # - ModelF7, ModelF8a: flex1 = w-reg
    # - ModelF6, ModelF8b: flex1 = a-reg  
    # - ModelF8: flex1 = w-reg, flex2 = a-reg
    has_regulator1_flexi = false  # w-reg flexi
    has_regulator2_flexi = false  # a-reg flexi
    
    if contains(flexi_label, "ModelF7") || contains(flexi_label, "ModelF8a")
        # flex1 = w-reg
        has_regulator1_flexi = haskey(flexi_p_repr, :flex1_params)
    elseif contains(flexi_label, "ModelF6") || contains(flexi_label, "ModelF8b") 
        # flex1 = a-reg
        has_regulator2_flexi = haskey(flexi_p_repr, :flex1_params)
    elseif contains(flexi_label, "ModelF8")
        # flex1 = w-reg, flex2 = a-reg
        has_regulator1_flexi = haskey(flexi_p_repr, :flex1_params)
        has_regulator2_flexi = haskey(flexi_p_repr, :flex2_params)
    end
    
    num_subplots = 0
    if has_regulator1_classical || has_regulator1_flexi
        num_subplots += 1
    end
    if has_regulator2_classical || has_regulator2_flexi
        num_subplots += 1
    end
    
    if num_subplots == 0
        # No regulators to plot
        ax = Makie.Axis(fig[1, 1]; axis_kwargs...)
        text!(ax, 0.5, 0.5; text="No regulator functions available for comparison", 
              align=(:center, :center))
        return fig
    end
    
    # Explicit layout handling for all cases
    # Case 1: Both w-reg and a-reg exist → 2 subplots (w-reg left, a-reg right)
    # Case 2: Only w-reg exists → 1 subplot (w-reg center)  
    # Case 3: Only a-reg exists → 1 subplot (a-reg center)
    
    has_w_reg = has_regulator1_classical || has_regulator1_flexi
    has_a_reg = has_regulator2_classical || has_regulator2_flexi
    
    # Create w-reg subplot if needed
    if has_w_reg
        # If both w-reg and a-reg exist, put w-reg on left; otherwise center it
        w_position = has_a_reg ? 1 : 1
        ax_w = Makie.Axis(fig[1, w_position]; 
                         xlabel="w", ylabel="Regulator function",
                         title="w → regulator")
        
        # Plot classical w-regulator
        if has_regulator1_classical
            w = range(0; stop=1.0, length=200)
            beta_mw = classical_p_all.p_classical.beta_mw
            nwm = classical_p_all.p_classical.nwm
            regulator = 1 ./ (1 .+ (abs.(beta_mw * w)) .^ nwm)
            lines!(ax_w, w, regulator; color=:blue, linewidth=2, linestyle=:solid,
                   label="$classical_label")
        end
        
        # Plot flexi w-regulator
        if has_regulator1_flexi
            # Determine which flex parameter to use for w-reg based on model
            if contains(flexi_label, "ModelF7") || contains(flexi_label, "ModelF8a") || contains(flexi_label, "ModelF8")
                # For these models, flex1 = w-reg
                w = range(0; stop=1.0, length=length(flexi_p_repr.flex1_params))
                flexi_values = [Model8.FlexiFunctions.evaluate_decompress(wi, flexi_p_repr.flex1_params) for wi in w]
            else
                error("Unexpected model for w-regulator flexi plotting: $flexi_label")
            end
            regulator = 1 ./ (1.0 .+ abs.(flexi_values))
            scatter!(ax_w, w, regulator; color=:red, markersize=4, alpha=0.7)
            lines!(ax_w, w, regulator; color=:red, linewidth=2, linestyle=:dash,
                   label="$flexi_label")
        end
        
        axislegend(ax_w; position=:rt)
    end
    
    # Create a-reg subplot if needed
    if has_a_reg
        # If both w-reg and a-reg exist, put a-reg on right; otherwise center it
        a_position = has_w_reg ? 2 : 1
        ax_a = Makie.Axis(fig[1, a_position];
                         xlabel="a", ylabel="Regulator function", 
                         title="a → regulator")
        
        # Plot classical a-regulator
        if has_regulator2_classical
            a = range(0; stop=1.0, length=200)
            beta_ma = classical_p_all.p_classical.beta_ma
            nam = classical_p_all.p_classical.nam
            regulator = 1 ./ (1 .+ (abs.(beta_ma * a)) .^ nam)
            lines!(ax_a, a, regulator; color=:blue, linewidth=2, linestyle=:solid,
                   label="$classical_label")
        end
        
        # Plot flexi a-regulator
        if has_regulator2_flexi
            # Determine which flex parameter to use for a-reg based on model
            if contains(flexi_label, "ModelF6") || contains(flexi_label, "ModelF8b")
                # For these models, flex1 = a-reg
                a = range(0; stop=1.0, length=length(flexi_p_repr.flex1_params))
                flexi_values = [Model8.FlexiFunctions.evaluate_decompress(ai, flexi_p_repr.flex1_params) for ai in a]
            elseif contains(flexi_label, "ModelF8")
                # For ModelF8, flex2 = a-reg
                a = range(0; stop=1.0, length=length(flexi_p_repr.flex2_params))
                flexi_values = [Model8.FlexiFunctions.evaluate_decompress(ai, flexi_p_repr.flex2_params) for ai in a]
            else
                error("Unexpected model for a-regulator flexi plotting: $flexi_label")
            end
            regulator = 1 ./ (1.0 .+ abs.(flexi_values))
            scatter!(ax_a, a, regulator; color=:red, markersize=4, alpha=0.7)
            lines!(ax_a, a, regulator; color=:red, linewidth=2, linestyle=:dash,
                   label="$flexi_label")
        end
        
        axislegend(ax_a; position=:rt)
    end
    
    return fig
end

"""
    plot_single_model_regulators(p_all_derepresented, p_repr;
                                 title=nothing, figure_size=(1000, 400))

Create a regulator plot for a single model (classical or flexi).

# Arguments
- `p_all_derepresented`: Derepresented parameters
- `p_repr`: Represented parameters (for accessing flex params if available)
- `title`: Optional title for the plot (default: auto-generated)
- `figure_size`: Figure size as (width, height) tuple (default: (1000, 400))

# Returns
- `fig`: Makie Figure object ready for display or saving

# Example
```julia
fig = Model8.plot_single_model_regulators(
    p_all_derepresented, best_p_repr;
    title="Regulator Functions"
)
```
"""
function plot_single_model_regulators(p_all_derepresented, p_repr;
                                     title=nothing,
                                     figure_size=(1000, 400))
    
    # Create figure
    fig = Figure(size=figure_size)
    
    # Handle title properly (avoid passing nothing to Makie)
    if !isnothing(title)
        fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)
    end
    
    # Check if we have classical regulator parameters
    has_classical_reg1 = haskey(p_all_derepresented.p_classical, :beta_mw)
    has_classical_reg2 = haskey(p_all_derepresented.p_classical, :beta_ma)
    
    # Check if we have flexi regulator parameters
    has_flexi_reg1 = haskey(p_repr, :flex1_params)
    has_flexi_reg2 = haskey(p_repr, :flex2_params)
    
    # Determine number of subplots needed
    num_regulators = 0
    if has_classical_reg1 || has_flexi_reg1
        num_regulators += 1
    end
    if has_classical_reg2 || has_flexi_reg2
        num_regulators += 1
    end
    
    if num_regulators == 0
        # No regulators to plot
        ax = Makie.Axis(fig[1, 1]; xlabel="Input", ylabel="Regulator function")
        text!(ax, 0.5, 0.5; text="No regulator functions available for this model", 
              align=(:center, :center))
        return fig
    end
    
    subplot_idx = 1
    
    # Plot regulator 1 (w-based) if available
    if has_classical_reg1 || has_flexi_reg1
        ax1 = Makie.Axis(fig[1, subplot_idx]; 
                        xlabel="w", ylabel="regulator 1",
                        title="w → regulator")
        
        # Classical regulator 1
        if has_classical_reg1
            w = range(0; stop=1.0, length=200)
            beta_mw = p_all_derepresented.p_classical.beta_mw
            nwm = p_all_derepresented.p_classical.nwm
            regulator = 1 ./ (1 .+ (abs.(beta_mw * w)) .^ nwm)
            lines!(ax1, w, regulator; color=:blue, linewidth=2,
                   label="Classical")
        end
        
        # Flexi regulator 1
        if has_flexi_reg1
            w = range(0; stop=1.0, length=length(p_repr.flex1_params))
            flexi1 = [Model8.FlexiFunctions.evaluate_decompress(wi, p_repr.flex1_params) for wi in w]
            regulator = 1 ./ (1.0 .+ abs.(flexi1))
            scatter!(ax1, w, regulator; color=:red, markersize=4, alpha=0.7)
            lines!(ax1, w, regulator; color=:red, linewidth=2,
                   label="Flexi")
        end
        
        if has_classical_reg1 && has_flexi_reg1
            axislegend(ax1; position=:rt)
        end
        subplot_idx += 1
    end
    
    # Plot regulator 2 (a-based) if available
    if has_classical_reg2 || has_flexi_reg2
        ax2 = Makie.Axis(fig[1, subplot_idx];
                        xlabel="a", ylabel="regulator 2", 
                        title="a → regulator")
        
        # Classical regulator 2
        if has_classical_reg2
            a = range(0; stop=1.0, length=200)
            beta_ma = p_all_derepresented.p_classical.beta_ma
            nam = p_all_derepresented.p_classical.nam
            regulator = 1 ./ (1 .+ (abs.(beta_ma * a)) .^ nam)
            lines!(ax2, a, regulator; color=:blue, linewidth=2,
                   label="Classical")
        end
        
        # Flexi regulator 2
        if has_flexi_reg2
            a = range(0; stop=1.0, length=length(p_repr.flex2_params))
            flexi2 = [Model8.FlexiFunctions.evaluate_decompress(ai, p_repr.flex2_params) for ai in a]
            regulator = 1 ./ (1.0 .+ abs.(flexi2))
            scatter!(ax2, a, regulator; color=:red, markersize=4, alpha=0.7)
            lines!(ax2, a, regulator; color=:red, linewidth=2,
                   label="Flexi")
        end
        
        if has_classical_reg2 && has_flexi_reg2
            axislegend(ax2; position=:rt)
        end
    end
    
    return fig
end