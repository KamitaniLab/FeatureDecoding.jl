"""DNN Feature decoding - feature prediction script"""

using Statistics

using BrainDecoder
using FastL2LiR
using HDF5
using Printf
using Glob
using Debugger


# Settings ###################################################################

# Brain data
brain_dir = "/home/kiss/data/fmri_shared/datasets/Deeprecon/fmriprep"
subjects = Dict(
    "AM" => "AM_ImageNetTest_volume_native.h5",
    # "ES" => "ES_ImageNetTest_volume_native.h5",
    # "FC" => "FC_ImageNetTest_volume_native.h5",
    # "JK" => "JK_ImageNetTest_volume_native.h5",
    # "JP" => "JP_ImageNetTest_volume_native.h5",
    # "KS" => "KS_ImageNetTest_volume_native.h5",
    # "TH" => "TH_ImageNetTest_volume_native.h5",
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

# DNN features
features_dir = "/home/nu/data/contents_shared/ImageNetTest/derivatives/features/default"
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

# Feature decoders
models_dir_root = "./results/feature_decoders/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100"

# Results directory
#results_dir_root = "/home/nu/data/contents_shared/ImageNetTraining/derivatives/feature_decoders/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100"
results_dir_root = "./results/decoded_features/deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100"


# Functions ##################################################################

function predict_features(model_dir, x; x_mean=[], x_norm=[], y_mean=[], y_norm=[], chunk_axis=0)

    # Normalize X
    x = (x .- x_mean) ./ x_norm

    model = load_model(model_dir, chunk_axis=chunk_axis)

    @show size(model.W)
    @show size(model.b)

    y = predict(model, x)

    # Denormalize Y
    y = y .* y_norm .+ y_mean

    return y
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

            # Feature prediction
            save_dir = joinpath(results_dir_root, "decoded_features", network, lay, sub, roi)
            if !isdir(save_dir)
                mkpath(save_dir)
            end
            # TODO: check existing files

            model_dir = joinpath(models_dir_root, network, lay, sub, roi, "model")

            x = brain_data.select(rois[roi])
            x_labels = brain_data.get_labels(label_key)

            x = Float32.(x)

            @show size(x)
            @show size(x_labels)

            # Average fMRI samples
            x_labels_unique = unique(x_labels)
            x_ave = cat([mean(x[findall(x -> x == xl, x_labels), :], dims=1)
                         for xl in unique(x_labels)]..., dims=1)

            @show size(x_ave)
            @show size(x_labels_unique)

            # Preprocessing parameters
            x_mean = load_array(joinpath(model_dir, "x_mean.mat"), "x_mean")
            x_norm = load_array(joinpath(model_dir, "x_norm.mat"), "x_norm")
            y_mean = load_array(joinpath(model_dir, "y_mean.mat"), "y_mean")
            y_norm = load_array(joinpath(model_dir, "y_norm.mat"), "y_norm")

            # Get sample index of Y
            y_index = [findall(x -> x == xl, y_labels)[1] for xl in x_labels_unique]

            # Call training routine
            @time y_pred = predict_features(
                model_dir, x_ave,
                x_mean=x_mean, x_norm=x_norm,
                y_mean=y_mean, y_norm=y_norm,
                chunk_axis=chunk_axis
            )

            @show size(y_pred)

            # Save decoded features
            y_idx = axes(y_pred)
            for (i, xl) in enumerate(x_labels_unique)
                save_file = joinpath(save_dir, xl * ".mat")
                save_array(save_file, "feat", y_pred[[i], y_idx[2:end]...])
            end
        end
    end
end

println("All done")
