function get_metrics(best_p_repr; learning_problem::LearningProblem{M}) where {M<:AbstractModel}

    # @show learning_problem
    # @show learning_problem.model

    p_derepresented = derepresent_all(best_p_repr, learning_problem.model)

    c24_normalized = get_freq_response(
        learning_problem.on_times, learning_problem.off_times, p_derepresented, learning_problem.model;
        continuous_pulses=learning_problem.continuous_pulses,
    )

    squared_residuals = (c24_normalized .- learning_problem.c24_data) .^ 2
    ssr_per_row = vec(sum(squared_residuals; dims=2))
    rms_per_row = sqrt.(ssr_per_row ./ size(squared_residuals, 2))
    total_ssr = sum(squared_residuals)
    total_rms = sqrt(total_ssr / (size(squared_residuals, 1) * size(squared_residuals, 2)))

    masked_squared_residuals = ((c24_normalized .- learning_problem.c24_data)[learning_problem.mask]) .^ 2
    masked_total_ssr = sum(masked_squared_residuals)
    masked_total_rms = sqrt(masked_total_ssr / sum(learning_problem.mask))

    unmasked_total_rms = NaN
    if sum(.!learning_problem.mask) != 0
        unmasked_squared_residuals = ((c24_normalized .- learning_problem.c24_data)[.!learning_problem.mask]) .^ 2
        unmasked_total_ssr = sum(unmasked_squared_residuals)
        unmasked_total_rms = sqrt(unmasked_total_ssr / sum(.!learning_problem.mask))
    end


    # Store results in a Dict
    metrics = Dict(
        "rms per on_time" => rms_per_row, 
        "rms for all on_times" => total_rms,
        # "masked rms per on_time" => masked_rms_per_row, 
        "masked rms" => masked_total_rms,
        "unmasked rms" => unmasked_total_rms,
        )

    return metrics
end

function print_metrics(metrics; output=stdout)
    println(output, "Metrics:")
    for (key, value) in metrics
        println(output, key * ": " * string(value))
        # for (key2, value2) in value
        #     println(output, "  ", key2)
        #     for (key3, value3) in value2
        #         println(output, "    ", key3, ": ", value3)
        #     end
        # end
    end
end
