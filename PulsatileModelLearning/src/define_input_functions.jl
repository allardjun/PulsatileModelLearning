# define a tophat function and pulsatile series

# function tophat_func(t,tstart,tstop,width)
#     return 1.0 ./((1+exp(-(t-tstart)/width)).*(1+exp(+(t-tstop)/width)))
# end

function tophat_func(t, tstart, tstop, width)
    return 0.5 * (tanh((t - tstart) / width) - tanh((t - tstop) / width))
end

function i_pulses(t, on_time, off_time; continuous_pulses=false)
    if continuous_pulses
        cumulative_duration = 1440.0 * 5
    else
        cumulative_duration = 360.0
    end
    
    if off_time > 0
        number_of_pulses = floor( cumulative_duration / on_time )
        period = on_time + off_time
        i = 0 # build up the signal by adding the pulses
        for current_pulse in 1:number_of_pulses
            i = i + tophat_func(t, (current_pulse - 1) * period, (current_pulse - 1) * period + on_time, 2.0)
        end
    else
        i = tophat_func(t, 0, cumulative_duration, 2.0)
    end

    return i
end


function i_heaviside(t, t_start)
    return tophat_func(t, t_start, 1440.0 * 5, 5.0) # turn off at effective infinity
end

function i_tophat(t, on_time)
    return tophat_func(t, 0, on_time, 5.0)
end
