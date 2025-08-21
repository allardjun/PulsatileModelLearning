
# Run it in the repl:
# Model8.get_data(save_data=true)

function get_data(; save_data=false)
    
    base_path = ""
    # base_path = JunTools.get_base_path("TCRPulsing")
    # @show base_path

    ## Read in xlsx from supplemeental material of Harris James 2020. 

    xf = XLSX.readxlsx(joinpath(base_path, "data/msb202010091-sup-0003-datasetev1.xlsx"))

    on_times = [15.0, 30.0, 60.0, 72.0, 90.0, 120.0] # Stored in header sentences in the xlsx. 

    data = Dict{String,Any}()

    ################################################################################
    ## CD69

    msh = xf["Figure 6E,F"]

    # Process time columns - they follow a pattern advancing by 5 columns
    time_cols = ["B", "G", "L", "Q", "V", "AA"]
    off_times = hcat([msh["$(col)4:$(col)17"] for col in time_cols]...)
    off_times = Matrix{Float64}(off_times)

    # Initialize result matrices
    cend = zeros((14, 6))
    cend_stdev = zeros((14, 6))

    # Process cend columns - they follow a pattern of 3 columns each, starting at C,H,M,R,W,AB
    start_cols = ["C", "H", "M", "R", "W", "AB"]
    for (i, col) in enumerate(start_cols)
        # For most columns, we can just get the next letter
        next_col = col == "AB" ? "AC" : string(Char(first(col) + 1))
        next_next_col = col == "AB" ? "AD" : string(Char(first(col) + 2))

        # Get the three adjacent columns
        cend1_1 = msh["$(col)23:$(col)36"]
        cend1_2 = msh["$(next_col)23:$(next_col)36"]
        cend1_3 = msh["$(next_next_col)23:$(next_next_col)36"]

        cend1 = Matrix{Float64}([cend1_1 cend1_2 cend1_3])
        cend_stdev[:, i] = std(cend1; dims=2)
        cend[:, i] = mean(cend1; dims=2)
    end

    # first index is on_time, second index is off_time.
    off_times_mat = off_times
    c24_mat = cend
    c24stdev_mat = cend_stdev

    # convert to list of lists
    off_times_lol = []
    for i in axes(on_times, 1)
        push!(off_times_lol, off_times_mat[:, i])
    end

    # convert to list of lists
    c24_lol = []
    c24stdev_lol = []
    for i in axes(on_times, 1)
        push!(c24_lol, c24_mat[:, i])
        push!(c24stdev_lol, c24stdev_mat[:, i])
    end

    # store in a Dict
    data["CD69"] = Dict(
        "on_times" => on_times,
        "off_times_mat" => off_times_mat,
        "c24_mat" => c24_mat,
        "c24stdev_mat" => c24stdev_mat,
        "off_times" => off_times_lol,
        "c24" => c24_lol,
        "c24stdev" => c24stdev_lol,
    )

    if save_data
        @save joinpath(base_path, "data/Harris_data_CD69.jld2") data
    end

    ################################################################################

end
