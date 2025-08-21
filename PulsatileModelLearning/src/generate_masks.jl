function generate_kfold_masks(shape::Tuple{Int, Int}, k::Int)
    n_total = prod(shape)  # total number of elements
    if n_total % k != 0
        error("Number of elements ($n_total) must be divisible by k ($k)")
    end
    fold_size = div(n_total, k)

    # Create a shuffled vector of all linear indices
    all_indices = shuffle(1:n_total)

    masks = Vector{BitMatrix}(undef, k)
    for i in 1:k
        # Indices to be masked as false (i.e., validation set)
        val_indices = all_indices[(fold_size * (i-1) + 1):(fold_size * i)]

        # Start with all true
        mask = trues(shape)
        # Set validation indices to false
        mask[val_indices] .= false

        masks[i] = mask
    end

    @save "assets/kfold_masks.jld2" masks

    return #masks

    # Example usage:
    #PulsatileModelLearning.generate_kfold_masks((6, 14), 7)

end



# Save the masks to a file

function generate_mask(config,c24_data)
    if config["mask_id"] != 0
        # load mask from file
        mask_file_name = "kfold_masks.jld2"
        @load joinpath("assets/", mask_file_name) masks
        mask = masks[config["mask_id"]]

        if size(mask) != size(c24_data)
            error("Mask size $(size(mask)) does not match c24_data size $(size(c24_data))")
        end
    else
        # construct mask that excludes some on_time,off_time pairs from SSR
        mask = trues(size(c24_data))
        for pair in config["mask_pairs"]
            mask[pair[1], pair[2]] = false
        end
    end
    return mask
end


# masks = load(joinpath("assets", "kfold_masks.jld2"), "masks")
# @show masks

function create_args(filename::String)

    mask_ids = 1:7
    flexi_dofs = [2,4,8,16,32,64,128,256]


    open(filename, "w") do io
        for a in mask_ids, b in flexi_dofs
            println(io, "--mask_id $a --flexi_dofs $b")
        end
    end
end

# Example usage:
# create_args("batch_scripts/args_cv.txt")