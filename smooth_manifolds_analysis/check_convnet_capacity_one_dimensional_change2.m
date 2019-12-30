function check_convnet_capacity_one_dimensional_change2(P, range_factor, N_SAMPLES, network_type, ...
    local_preprocessing, run_id, epoch, D, center_norm_factor, samples_jump, ...
    features_type, N_OBJECTS, use_raw_images, random_labeling_type, layers_grouping_level)
% Calculate capacity for the given manifolds
% This version uses an adaptive number of samples and does not collect manifold properties 
% to minimize computational cost.
if nargin < 5
    local_preprocessing = 0; % 0- none, 1- orthogonalize centers, 2- random centers, 3- permuted manifold
end
if nargin < 6
    run_id = 0;
end
if nargin < 7
    epoch = nan;
end
if nargin < 8
    D = nan;
end
if nargin < 9
	center_norm_factor = 1;
end
if nargin < 10
    samples_jump = 1;
end
if nargin < 11
    features_type = 2; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
end
if nargin < 12
    N_OBJECTS = P;
else
    assert(N_OBJECTS <= P);    
end
if nargin < 13
    use_raw_images = false;
end
if nargin < 14
    random_labeling_type = 1; % 0- binary iid, 1- balanced, 2- sparse
end
if nargin < 15
    if network_type >= 2 && network_type <= 4
        layers_grouping_level = 2;
    else
        layers_grouping_level = 0;
    end	
end

global_preprocessing = 0; % 0- none, 1- znorm, 2- whitening, 3- centers decorelation
input_suffix = '';

% Expected precision ep=sqrt(pq/n) so that for p=q=0.5 ep=0.05 yields n=100
EXPECTED_PRECISION = 0.05; 
max_samples = 100;
precision = 1;

[network_name, N_LAYERS, ~, layer_names, ~, ~, ~, ~, ~, ~, ~, layers, ~] = ...
    load_network_metadata(network_type, layers_grouping_level, epoch);
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
if ~isnan(epoch) && isempty(input_suffix)
    input_suffix = sprintf('_epoch%d', epoch);
end

direction_names = {'x-translation', 'y-translation', 'x-scale', 'y-scale', 'x-shear', 'y-shear', 'rotation'};
N_DIRECTIONS = length(direction_names);

function [out_name, common_corr_name] = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, D, center_norm_factor, samples_jump, features_type, run_id)
    prefix = sprintf('check_%s_capacity_one_dimensional_change2', network_name);
    common_corr_prefix = sprintf('check_%s_covariance_low_rank_approx_optimal_K', network_name);
    if IMAGENET_IMAGE_SIZE ~= 64
        prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
        common_corr_prefix = sprintf('%s_%dpx', common_corr_prefix, IMAGENET_IMAGE_SIZE);
    end
    if features_type == 2
        prefix = sprintf('%s_projected', prefix);
    else
        assert(features_type == 0, 'Unknown type of features to use: %d', features_type);
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
    if ~isnan(D)
        suffix = sprintf('%s_D%d', suffix, D);
    end
    if center_norm_factor ~= 1
        suffix = sprintf('%s_f%1.2f', suffix, center_norm_factor);
    end
    if samples_jump ~= 1
        suffix = sprintf('%s_s%d', suffix, samples_jump);
    end
    if run_id > 0
        suffix = sprintf('%s_%d', suffix, run_id);
    end

    common_corr_output_suffix = '_nullspace_slow';
    if range_factor < 0.1
        out_name = sprintf('%s%s_range%f_P%d_M%d%s.mat', prefix, input_suffix, range_factor, N_OBJECTS, N_SAMPLES, suffix);
        common_corr_name = sprintf('%s%s_range%f_P%d_M%d%s.mat', common_corr_prefix, input_suffix, range_factor, N_OBJECTS, N_SAMPLES, common_corr_output_suffix);
    else
        out_name = sprintf('%s%s_range%1.1f_P%d_M%d%s.mat', prefix, input_suffix, range_factor, N_OBJECTS, N_SAMPLES, suffix);
        common_corr_name = sprintf('%s%s_range%1.1f_P%d_M%d%s.mat', common_corr_prefix, input_suffix, range_factor, N_OBJECTS, N_SAMPLES, common_corr_output_suffix);
    end
end

% Result variables
capacity_results            = nan(N_DIRECTIONS, N_LAYERS, 1);
separability_results        = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS);
neuron_samples_used_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS);
n_support_vectors_results   = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS, N_OBJECTS);

if length(run_id) > 1
    run_ids = run_id;
    out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, D, center_norm_factor, samples_jump, features_type, 0);
    fprintf('Results saved to %s\n', out_name);

    missing = 0;
    for run_id=run_ids
        run_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, D, center_norm_factor, samples_jump, features_type, run_id);
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
        neuron_samples_used_results(param_id, l, :) = run_file.neuron_samples_used_results(param_id, l, :);
        if isprop(run_file, 'n_support_vectors_results')
            n_support_vectors_results(param_id, l, :, :) = run_file.n_support_vectors_results(param_id, l, :, :);
        end
    end
    if missing > 0 
        fprintf('Done with %d missing values\n', missing);
    end

    % Save results
    save(out_name, 'capacity_results', 'separability_results', 'neuron_samples_used_results', 'n_support_vectors_results', ...
        'EXPECTED_PRECISION', 'max_samples', 'precision', 'features_type', '-v7.3');
    return;
end
        
out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, D, center_norm_factor, samples_jump, features_type, 0);
if exist(out_name, 'file')
    fprintf('Loading existing collected results\n');
    load(out_name);
end

[out_name, common_corr_name] = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, local_preprocessing, D, center_norm_factor, samples_jump, features_type, run_id);

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
        
        if (local_preprocessing == 0) && isnan(D) && (center_norm_factor == 1)
            current_tuning_function = full_tuning_function(nzIndices,:,:);
        else
            assert(exist(common_corr_name, 'file')>0, 'Cannot find %s', common_corr_name);
            fprintf('Loading common components from %s\n', common_corr_name);
            common_corr_file = matfile(common_corr_name);
            Kopt = common_corr_file.best_K_results(param_id, l);
            assert(isnan(common_corr_file.effective_n_results(param_id, l)) || N == common_corr_file.effective_n_results(param_id, l), 'Effective N=%d differ from that of common correlations file (%d)', N, common_corr_file.effective_n_results(param_id, l));
            Vk = reshape(common_corr_file.common_subspace_results(param_id, l, 1:N, 1:Kopt), [N, Kopt]);
            current_tuning_function = calc_low_dimension_manifolds_preserving_correlations(full_tuning_function(nzIndices,:,:), ...
                Vk, D, center_norm_factor, local_preprocessing);
        end
        if samples_jump == 0 
            % Use a single sample
            assert(mod(N_SAMPLES,2) == 1, 'N_SAMPLES must be odd');
            current_tuning_function = current_tuning_function(:,(N_SAMPLES+1)/2,:);
        else
            assert(mod(N_SAMPLES-1, samples_jump) == 0, 'samples_jump must divide N_SAMPLES-1');
            current_tuning_function = current_tuning_function(:,1:samples_jump:N_SAMPLES,:);
        end

        [capacity, separability, ~, n_neuron_samples_used, n_support_vectors] = ...
            check_binary_dichotomies_capacity2(current_tuning_function, EXPECTED_PRECISION, true, random_labeling_type, ...
                precision, max_samples, global_preprocessing, features_type);
        
        capacity_results(param_id, l) = capacity;
        separability_results(param_id, l, 1:N) = separability;
        neuron_samples_used_results(param_id, l, 1:N) = n_neuron_samples_used;
        n_support_vectors_results(param_id, l, 1:N, :) = n_support_vectors;
        save(out_name, 'capacity_results', 'separability_results', 'neuron_samples_used_results', 'n_support_vectors_results', ...
            'EXPECTED_PRECISION', 'max_samples', 'precision', 'features_type', '-v7.3');
    end
end
fprintf('Done. (took %1.1f hours)\n', toc(T)/3600.);
end
