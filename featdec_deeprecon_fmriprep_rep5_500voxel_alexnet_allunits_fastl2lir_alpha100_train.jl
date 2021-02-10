"""DNN Feature decoding - decoders training script"""

using Statistics

using BrainDecoder
using FastL2LiR
using HDF5
using Printf
using Debugger

# Settings ###################################################################

# Brain data
brain_dir = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
subjects = Dict(
    "AM" => "AM_ImageNetTraining_volume_native.h5",
    # "ES" => "ES_ImageNetTraining_volume_native.h5",
    # "FC" => "FC_ImageNetTraining_volume_native.h5",
    # "JK" => "JK_ImageNetTraining_volume_native.h5",
    # "JP" => "JP_ImageNetTraining_volume_native.h5",
    # "KS" => "KS_ImageNetTraining_volume_native.h5",
    # "TH" => "TH_ImageNetTraining_volume_native.h5",
)

label_key = "stimulus_name"

rois = Dict(
    "VC"  => "ROI_VC",
    # "LVC" => "ROI_LVC",
    # "HVC" => "ROI_HVC",
    # "V1"  => "ROI_V1",
    # "V2"  => "ROI_V2",
    # "V3"  => "ROI_V3",
    # "V4"  => "ROI_hV4",
    # "LOC" => "ROI_LOC",
    # "FFA" => "ROI_FFA",
    # "PPA" => "ROI_PPA",
)

num_voxels = Dict(
    "VC"  => 500,
    "LVC" => 500,
    "HVC" => 500,
    "V1"  => 500,
    "V2"  => 500,
    "V3"  => 500,
    "V4"  => 500,
    "LOC" => 500,
    "FFA" => 500,
    "PPA" => 500,
)

# DNN features
features_dir = "/home/nu/data/contents_shared/ImageNetTraining/derivatives/features/default"
network = "caffe/bvlc_alexnet"
layers = [
    "conv1", #"relu1", "pool1",
    "conv2", #"relu2", "pool2",
    "conv3", #"relu3",
    "conv4", #"relu4",
    "conv5", #"relu5", "pool5",
    "fc6", #"relu6",
    "fc7", #"relu7",
    "fc8",
]

chunk_axis = 0
# Note that Julia's indexing is one-based.

# Model parameters
alpha = 100.0

# Results directory
#results_dir_root = "/home/nu/data/contents_shared/ImageNetTraining/derivatives/feature_decoders/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100"
results_dir_root = "./results/feature_decoders/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100"


# Functions ##################################################################

function train_decoder(x, y, alpha, n_voxels; x_mean=[], x_norm=[], y_mean=[], y_norm=[], y_index=[], chunk_axis=0, save_dir="./model")

    function train_decoder_chunk(_x, _y)
        if length(y_index) != 0
            _y = _y[y_index, fill(:, ndims(_y) - 1)...]
        end

        @time model = fit(_x, _y, Float32(alpha), n_voxels)

        return model
    end

    if length(x_mean) != 0 && length(x_norm) != 0
        x_n = (x .- x_mean) ./ x_norm
    else
        x_n = x
    end

    if length(y_mean) != 0 && length(y_norm) != 0
        y_n = (y .- y_mean) ./ y_norm
    else
        y_n = y
    end

    if chunk_axis != 0 && ndims(y) > 3
        w_dir = joinpath(save_dir, "W")
        b_dir = joinpath(save_dir, "b")
        if !isdir(save_dir)
            mkpath(save_dir)
        end
        if !isdir(w_dir)
            mkpath(w_dir)
        end
        if !isdir(b_dir)
            mkpath(b_dir)
        end

        # Chunking
        n_chunks = size(y)[chunk_axis]

        for i in 1:n_chunks
            println(@sprintf("Chunk %d", i - 1))
            idx = [(if n == chunk_axis fill(i, 1) else : end) for n in 1:ndims(y_n)]
            y_n_chunk = y_n[idx...]
            @show size(y_n_chunk)
            model = train_decoder_chunk(x_n, y_n_chunk)

            w_file = joinpath(w_dir, @sprintf("%08d", i - 1) * ".mat")
            b_file = joinpath(b_dir, @sprintf("%08d", i - 1) * ".mat")

            save_array(w_file, "W", model.W, sparse=true)
            save_array(b_file, "b", model.b, sparse=true)
        end

    else
        # No chunking
        model = train_decoder_chunk(x_n, y_n)

        w_file = joinpath(save_dir, "W.mat")
        b_file = joinpath(save_dir, "b.mat")

        save_array(w_file, "W", model.W, sparse=true)
        save_array(b_file, "b", model.b, sparse=true)
    end
end

# Main #######################################################################

# Load data ------------------------------------------------------------------
features = Features(joinpath(features_dir, network))

# Analysis loop --------------------------------------------------------------
println("----------------------------------------")
println("Analysis loop")

for lay in layers

    # Load DNN features
    println("Loading features: " * lay)
    y = features.get_features(lay)  # TODO: speed-up
    y_labels = features.labels

    y = Float32.(y)

    @show size(y)
    @show size(y_labels)

    for sub in keys(subjects)

        # Load fMRI data
        brain_data_file = joinpath(brain_dir, subjects[sub])
        println("Loading " * brain_data_file)
        brain_data = BData(brain_data_file)

        for roi in keys(rois)
            println("Layer:   " * lay)
            println("Subject: " * sub)
            println("ROI:     " * roi)

            save_dir = joinpath(results_dir_root, network, lay, sub, roi, "model")
            if !isdir(save_dir)
                mkpath(save_dir)
            end
            # TODO: check existing files

            x = brain_data.select(rois[roi])
            x_labels = brain_data.get_labels(label_key)

            x = Float32.(x)

            @show size(x)
            @show size(x_labels)

            n_voxels = num_voxels[roi]

            x_mean = mean(x, dims=1)
            x_norm = std(x, corrected=true, dims=1)
            y_mean = mean(y, dims=1)
            y_norm = std(y, corrected=true, dims=1)

            save_array(joinpath(save_dir, "x_mean.mat"), "x_mean", x_mean)
            save_array(joinpath(save_dir, "x_norm.mat"), "x_norm", x_norm)
            save_array(joinpath(save_dir, "y_mean.mat"), "y_mean", y_mean, sparse=true)
            save_array(joinpath(save_dir, "y_norm.mat"), "y_norm", y_norm)

            # Get sample index of Y
            y_index = [findall(x -> x == xl, y_labels)[1] for xl in x_labels]

            # Call training routine
            result = train_decoder(
                x, y, alpha, n_voxels,
                x_mean=x_mean, x_norm=x_norm,
                y_mean=y_mean, y_norm=y_norm,
                y_index=y_index,
                chunk_axis=chunk_axis,
                save_dir=save_dir
                )
        end
    end
end

println("All done")
