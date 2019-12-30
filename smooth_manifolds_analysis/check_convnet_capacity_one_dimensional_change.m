function check_convnet_capacity_one_dimensional_change(P, range_factor, N_SAMPLES, network_type, global_preprocessing, local_preprocessing, input_suffix, run_id, random_labeling_type, layers_grouping_level, use_half_samples, features_type, N_OBJECTS, use_raw_images)
if nargin < 5
    global_preprocessing = 0; % 0- none, 1- znorm, 2- whitening, 3- centers decorelation
end
if nargin < 6
    local_preprocessing = 0; % 0- none, 1- orthogonalize centers, 2- random centers, 3- permuted manifold
end
if nargin < 7
    input_suffix = '';
end
if nargin < 8
    run_id = 0;
end
if nargin < 9
    random_labeling_type = 1; % 0- binary iid, 1- balanced, 2- sparse
end
if nargin < 10
    if network_type == 2
        layers_grouping_level = 2;
    else
        layers_grouping_level = 0;
    end
end
if nargin < 11
    use_half_samples = false;
end
if nargin < 12
    features_type = 2; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
end
if nargin < 13
    N_OBJECTS = P;
else
    assert(N_OBJECTS <= P);
end
if nargin < 14
	use_raw_images = false;
end
precise_mode = false;

if precise_mode
    N_NEURON_SAMPLES = 1001;
else
    N_NEURON_SAMPLES = 41;
end
N_DICHOTOMIES = 1;
max_samples = 100;
precision = 1;

[network_name, N_LAYERS, ~, layer_names, ~, ~, ~, ~, ~, ~, ~, layers, ~] = ...
    load_network_metadata(network_type, layers_grouping_level);
ENABLED_LAYERS = zeros(1, N_LAYERS); ENABLED_LAYERS(layers) = 1;

global N_HMAX_FEATURES;
global IMAGENET_IMAGE_SIZE;
if use_raw_images
    N_NEURONS = IMAGENET_IMAGE_SIZE*IMAGENET_IMAGE_SIZE*3;
    network_name = 'imagenet';
else
    N_NEURONS = N_HMAX_FEATURES;
    assert(~isempty(N_NEURONS), 'Run init_imagenet');
end

input_prefix = sprintf('%s/generate_%s_one_dimensional_change', network_name, network_name);
if IMAGENET_IMAGE_SIZE ~= 64
    input_prefix = sprintf('%s_%dpx', input_prefix, IMAGENET_IMAGE_SIZE);
end

direction_names = {'x-translation', 'y-translation', 'x-scale', 'y-scale', 'x-shear', 'y-shear', 'rotation'};
N_DIRECTIONS = length(direction_names);

function out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, global_preprocessing, input_suffix, use_half_samples, features_type, run_id)
    prefix = sprintf('check_%s_capacity_one_dimensional_change', network_name);
    if IMAGENET_IMAGE_SIZE ~= 64
        prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
    end
    if features_type == 2
        prefix = sprintf('%s_projected', prefix);
    else
        assert(features_type == 0, 'Unknown type of features to use: %d', features_type);
    end
    if precise_mode
        prefix = sprintf('%s_precise', prefix);
    end
    suffix = '';
    if random_labeling_type == 1
        suffix = [suffix, '_balanced'];
    elseif random_labeling_type == 2
        suffix = [suffix, '_sparse'];
    end
    if global_preprocessing == 1
        suffix = [suffix, '_znorm'];
    elseif global_preprocessing == 2
        suffix = [suffix, '_whiten'];
    elseif global_preprocessing == 3
        suffix = [suffix, '_centers_whiten'];
    end
    if local_preprocessing == 0
	elseif local_preprocessing == 1
        suffix = [suffix, '_orth'];
    elseif local_preprocessing == 2
        suffix = [suffix, '_centers_random'];
    elseif local_preprocessing == 3
        suffix = [suffix, '_manifold_random'];
    elseif local_preprocessing == 4
        suffix = [suffix, '_manifold_random_uniform_centers'];
    elseif local_preprocessing == 5
        suffix = [suffix, '_axes_random'];
    else
        assert(local_preprocessing == 7);
        suffix = [suffix, '_permute_random'];
    end
    if use_half_samples
        suffix = [suffix, '_half'];
    end
    if run_id > 0
        suffix = sprintf('%s_%d', suffix, run_id);
    end
    
    if range_factor < 0.1
        out_name = sprintf('%s_range%f_P%d_M%d%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, suffix, input_suffix);
    else
        out_name = sprintf('%s_range%1.1f_P%d_M%d%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, suffix, input_suffix);
    end
end

% Result variables
capacity_results            = nan(N_DIRECTIONS, N_LAYERS, 1);
separability_results        = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS);
if precise_mode
    radius_results              = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    mean_half_width_results     = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    mean_argmax_norm_results    = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    mean_half_width2_results    = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    mean_argmax_norm2_results   = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    effective_dimension_results = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    effective_dimension2_results = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    alphac_hat_results          = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
    features_used_results       = nan(N_DIRECTIONS, N_LAYERS, 1, N_NEURONS);
    labels_used_results         = nan(N_DIRECTIONS, N_LAYERS, 1, N_OBJECTS);
else
    radius_results              = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    mean_half_width_results     = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm_results    = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    mean_half_width2_results    = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm2_results   = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension2_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    alphac_hat_results          = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
    features_used_results       = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_NEURONS);
    labels_used_results         = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
end

if length(run_id) > 1
    run_ids = run_id;
    out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, global_preprocessing, input_suffix, use_half_samples, features_type, 0);
    fprintf('Results saved to %s\n', out_name);

    missing = 0;
    for run_id=run_ids
        run_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, global_preprocessing, input_suffix, use_half_samples, features_type, run_id);
        fprintf('Results loaded from %s\n', run_name);
        [param_id, l]=ind2sub([N_DIRECTIONS,N_LAYERS], run_id);
        assert(~ENABLED_LAYERS(l) || exist(run_name, 'file')>0);
        if ~exist(run_name, 'file')
            fprintf('Skipping missing file: %s\n', run_name);
            missing = missing + ENABLED_LAYERS(l);
            continue;
        end
        run_file = matfile(run_name);
        % Fill results
        capacity_results(param_id, l) = run_file.capacity_results(param_id, l);
        separability_results(param_id, l, :) = run_file.separability_results(param_id, l, :);
        radius_results(param_id, l, :, :)  = run_file.radius_results(param_id, l, :, :);
        mean_half_width_results(param_id, l, :, :)  = run_file.mean_half_width_results(param_id, l, :, :);
        mean_argmax_norm_results(param_id, l, :, :)  = run_file.mean_argmax_norm_results(param_id, l, :, :);
        mean_half_width2_results(param_id, l, :, :)  = run_file.mean_half_width2_results(param_id, l, :, :);
        mean_argmax_norm2_results(param_id, l, :, :)  = run_file.mean_argmax_norm2_results(param_id, l, :, :);
        effective_dimension_results(param_id, l, :, :)  = run_file.effective_dimension_results(param_id, l, :, :);
        if isprop(run_file, 'effective_dimension2_results')
            effective_dimension2_results(param_id, l, :, :)  = run_file.effective_dimension2_results(param_id, l, :, :);
        end
        alphac_hat_results(param_id, l, :, :)  = run_file.alphac_hat_results(param_id, l, :, :);
        features_used_results(param_id, l, :, :)  = run_file.features_used_results(param_id, l, :, :);
        labels_used_results(param_id, l, :, :)  = run_file.labels_used_results(param_id, l, :, :);
    end
    if missing > 0 
        fprintf('Done with %d missing values\n', missing);
    end

    % Save results
    save(out_name, 'capacity_results', 'separability_results', 'radius_results', ...
        'mean_half_width_results', 'mean_argmax_norm_results', 'mean_half_width2_results', 'mean_argmax_norm2_results', ...
        'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
        'features_used_results', 'labels_used_results', '-v7.3');
    return;
end
        
out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, global_preprocessing, input_suffix, use_half_samples, features_type, 0);
if exist(out_name, 'file')
    fprintf('Loading existing collected results\n');
    load(out_name);
end

out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, global_preprocessing, input_suffix, use_half_samples, features_type, run_id);
fprintf('Results saved to %s\n', out_name);

if exist(out_name, 'file')
    fprintf('Loading existing results\n');
    load(out_name);
end

if run_id == 0
    RUN_DIRECTIONS = 1:N_DIRECTIONS;
    RUN_LAYERS = 1:N_LAYERS;
else
    [param_id, l]=ind2sub([N_DIRECTIONS,N_LAYERS], run_id);
    RUN_DIRECTIONS = param_id;
    RUN_LAYERS = l;
end

T=tic;
for param_id=RUN_DIRECTIONS
    for l=RUN_LAYERS
        if use_raw_images
            assert(l == 1, 'Images can be used only with the first layer');
            layer_name = 'images';
        else
            layer_name = layer_names{l};
        end
        if ~ENABLED_LAYERS(l)
            fprintf('Skipping disabled layer %s\n', layer_name);
            continue;
        end 

        %a = mean_argmax_norm2_results(param_id, l, :, :);
        %if isfinite(capacity_results(param_id, l)) && all(isfinite(a(:)))
        if isfinite(capacity_results(param_id, l))
            fprintf('Skipping existing %s %s\n', direction_names{param_id}, layer_name);
            continue;
        end
        fprintf('Working on %s %s\n', direction_names{param_id}, layer_name);

        if range_factor < 0.1
            in_name = sprintf('%s_range%f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_name, input_suffix);
        else
            in_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_name, input_suffix);
        end
        in_file = matfile(in_name);
        
        tic;
        if use_raw_images
            % tuning_function = zeros(N_DIRECTIONS, N_OBJECTS, N_SAMPLES, IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_SIZE, 3, 'uint8');
            full_tuning_function = double(permute(reshape(in_file.tuning_function(param_id, 1:N_OBJECTS, :, :, :, :), [N_OBJECTS, N_SAMPLES, N_NEURONS]), [3, 2, 1]));
            assert(all(size(full_tuning_function) == [N_NEURONS, N_SAMPLES, N_OBJECTS]));
        else
            full_tuning_function = double(permute(squeeze(in_file.tuning_function(param_id, 1:N_OBJECTS, :, :)), [3, 2, 1]));
            assert(all(size(full_tuning_function) == [N_NEURONS, N_SAMPLES, N_OBJECTS]));
        end
        mean_squeare_firing_rate = mean(mean(full_tuning_function.^2, 3),2);
        nzIndices = find(mean_squeare_firing_rate > 0);
        N = length(nzIndices);
        fprintf('Loaded data (took %1.1f sec)\n', toc);
        
        if local_preprocessing == 0
            properties_type = 1;
            current_tuning_function = full_tuning_function(nzIndices,:,:);
        else
            properties_type = 0;
            minN = min(N_SAMPLES, N);
            current_tuning_function = calc_low_dimension_manifold(full_tuning_function(nzIndices,:,:), minN, local_preprocessing);
        end
        if use_half_samples
            current_tuning_function = current_tuning_function(:,1:2:N_SAMPLES,:);
        end

        [capacity, separability, ~, radius, mean_half_width1, mean_argmax_norm1, ...
            mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, ...
            features_used, labels_used] = ...
            check_binary_dichotomies_capacity(current_tuning_function, N_NEURON_SAMPLES, N_DICHOTOMIES, true, random_labeling_type, ...
                precision, max_samples, global_preprocessing, features_type, [], properties_type);
        
        capacity_results(param_id, l) = capacity;
        separability_results(param_id, l, 1:N) = separability;
        if precise_mode
            radius_results(param_id, l, :, :) = mean(radius, 1);
            mean_half_width_results(param_id, l, :, :) = mean(mean_half_width1,1);
            mean_argmax_norm_results(param_id, l, :, :) = mean(mean_argmax_norm1,1);
            mean_half_width2_results(param_id, l, :, :) = mean(mean_half_width2, 1);
            mean_argmax_norm2_results(param_id, l, :, :) = mean(mean_argmax_norm2, 1);
            effective_dimension_results(param_id, l, :, :) = mean(effective_dimension, 1);
            effective_dimension2_results(param_id, l, :, :) = mean(effective_dimension2, 1);
            alphac_hat_results(param_id, l, :, :) = mean(alphac_hat, 1);
        else
            radius_results(param_id, l, :, :) = radius;
            mean_half_width_results(param_id, l, :, :) = mean_half_width1;
            mean_argmax_norm_results(param_id, l, :, :) = mean_argmax_norm1;
            mean_half_width2_results(param_id, l, :, :) = mean_half_width2;
            mean_argmax_norm2_results(param_id, l, :, :) = mean_argmax_norm2;
            effective_dimension_results(param_id, l, :, :) = effective_dimension;
            effective_dimension2_results(param_id, l, :, :) = effective_dimension2;
            alphac_hat_results(param_id, l, :, :) = alphac_hat;
            if ~isempty(features_used)
                features_used_results(param_id, l, :, 1:capacity) = features_used;
            end
            labels_used_results(param_id, l, :, :) = labels_used;
        end
        
        save(out_name, 'capacity_results', 'separability_results', 'radius_results', ...
            'mean_half_width_results', 'mean_argmax_norm_results', 'mean_half_width2_results', 'mean_argmax_norm2_results', ...
            'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
            'features_used_results', 'labels_used_results', '-v7.3');
    end
end
fprintf('Done. (took %1.1f hours)\n', toc(T)/3600.);
end
