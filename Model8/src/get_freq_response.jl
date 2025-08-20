
# Optimized version with fast path for core usage
function get_freq_response(
    on_times, off_times, p_derepresented, model::M; make_timeseries=false, get_extrema=false, continuous_pulses=false
) where {M<:AbstractModel}
    
    # Fast path for core usage (performance critical - used in optimization loops)
    if !make_timeseries && !get_extrema
        return _get_freq_response_fast_path(on_times, off_times, p_derepresented, model; continuous_pulses)
    end
    
    # Legacy compatibility path for analysis features
    return _get_freq_response_with_analysis(on_times, off_times, p_derepresented, model; 
                                           make_timeseries, get_extrema, continuous_pulses)
end

# Fast path implementation - identical to original for performance
function _get_freq_response_fast_path(on_times, off_times, p_derepresented, model::M; continuous_pulses=false) where {M<:AbstractModel}
    c24_normalized = zeros(Real, length(on_times), length(off_times[1]))
    c24_unnormalized = zeros(Real, length(on_times), length(off_times[1]))

    for (idx_on_time, on_time) in enumerate(on_times)
        for (idx_off_time, off_time) in enumerate(off_times[idx_on_time])
            i_func = t -> i_pulses(t, on_time, off_time; continuous_pulses=continuous_pulses)
            rhs = Model8.make_rhs(i_func, model)
            prob = ODEProblem(rhs, model.u0, [0.0, 24 * 60.0])
            ode_solution = solve(prob, Vern9(); p=p_derepresented, abstol=1e-8, reltol=1e-8)

            c24_unnormalized[idx_on_time, idx_off_time] = ode_solution[4, end] # c24
            c24_normalized[idx_on_time, idx_off_time] =
                c24_unnormalized[idx_on_time, idx_off_time] / c24_unnormalized[idx_on_time, 1]
        end
    end

    return c24_normalized
end

# Legacy analysis implementation - maintains exact same behavior and return types
function _get_freq_response_with_analysis(on_times, off_times, p_derepresented, model::M; 
                                         make_timeseries=false, get_extrema=false, continuous_pulses=false) where {M<:AbstractModel}
    c24_normalized = zeros(Real, length(on_times), length(off_times[1]))
    c24_unnormalized = zeros(Real, length(on_times), length(off_times[1]))

    if make_timeseries
        figs = []
    end
    if get_extrema
        maxima = zeros(Real, length(on_times), length(off_times[1]), 4)
        minima = zeros(Real, length(on_times), length(off_times[1]), 4)
        captured_warnings = []
    end

    for (idx_on_time, on_time) in enumerate(on_times)
        for (idx_off_time, off_time) in enumerate(off_times[idx_on_time])
            i_func = t -> i_pulses(t, on_time, off_time; continuous_pulses=continuous_pulses)
            rhs = Model8.make_rhs(i_func, model)
            prob = ODEProblem(rhs, model.u0, [0.0, 24 * 60.0])

            ode_solution = solve(prob, Vern9(); p=p_derepresented, abstol=1e-8, reltol=1e-8)

            # plot every time series
            if make_timeseries || get_extrema
                u = Array(ode_solution)
                if make_timeseries
                    t = ode_solution.t
                    fig = create_timeseries_figure(t, u, i_func, p_derepresented)
                    push!(figs, fig)
                end
                if get_extrema
                    maxima[idx_on_time, idx_off_time, :] = [maximum(u[i, :]) for i in 1:4]
                    minima[idx_on_time, idx_off_time, :] = [minimum(u[i, :]) for i in 1:4]
                end
            end

            c24_unnormalized[idx_on_time, idx_off_time] = ode_solution[4, end] # c24
            c24_normalized[idx_on_time, idx_off_time] =
                c24_unnormalized[idx_on_time, idx_off_time] / c24_unnormalized[idx_on_time, 1]
        end
    end

    # Return early if no analysis is requested
    if !make_timeseries && !get_extrema
        return c24_normalized
    end

    if make_timeseries || get_extrema
        analysis = Dict()
        if make_timeseries
            analysis["figs"] = figs
        end
        if get_extrema
            analysis["maxima"] = maxima
            analysis["minima"] = minima
        end
        analysis["captured_warnings"] = captured_warnings
        return c24_normalized, analysis
    end
end