## ---------------------------------------------------------------------------
## Define the universal function 
## ---------------------------------------------------------------------------

module FlexiFunctions

# using Zygote

# export evaluate_decompress, evaluate_decompress_O2, generate_flexi_ig

# function evaluate_decompress(x::Real, params)
#     thetas = [0; cumsum(params .^ 2)]

#     N = length(thetas)
#     N_intervals = N-1

#     i = Zygote.@ignore min(floor(Int, N_intervals * x) + 1, N_intervals)
#     j = Zygote.@ignore i + 1

#     x_i = Zygote.@ignore (i - 1) / N_intervals
#     x_j = Zygote.@ignore i / N_intervals

#     return thetas[i] + (x - x_i) * (thetas[j] - thetas[i]) / (x_j - x_i)
# end

@inline function evaluate_decompress(x::Real, params::SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true})
    N_intervals = length(params)

    # Early exit for edge cases
    if x <= 0.0
        return 0.0
    elseif x >= 1.0
        # Return the complete sum of squares - could be precomputed if called repeatedly with x=1
        theta_N = 0.0
        @fastmath @inbounds for k in 1:N_intervals
            theta_N += params[k] * params[k]
        end
        return theta_N
    end

    # Find the interval - avoid division when possible
    scaled_x = x * N_intervals
    i = min(floor(Int, scaled_x) + 1, N_intervals)
    frac = scaled_x - (i - 1) # Fractional part for interpolation

    # Compute theta_i - the sum of squares up to but not including i
    theta_i = 0.0
    @fastmath @inbounds @simd for k in 1:(i - 1)
        theta_i += params[k] * params[k]
    end

    # Compute theta_j - add the square of the current parameter
    param_i_squared = params[i] * params[i]
    theta_j = theta_i + param_i_squared

    # Final interpolation
    return @fastmath theta_i + frac * param_i_squared
end

function generate_flexi_ig(flexi_dofs; beta=1.0, n=1.0)
    if n == 1.0
        return fill(sqrt(beta / flexi_dofs), flexi_dofs)
    else
        # For f(x) = (β·x)^n, we need to carefully select parameters
        # that will create the right cumulative sums

        # Calculate the sequence of values we need at each interval boundary
        x_points = collect(range(0, 1; length=flexi_dofs + 1))
        y_values = (beta .* x_points) .^ n

        # Calculate the differences that will create these values when squared and cumulatively summed
        differences = diff(y_values)
        params = sqrt.(differences)

        return params
    end
end

end # end of module FlexiFunctions
