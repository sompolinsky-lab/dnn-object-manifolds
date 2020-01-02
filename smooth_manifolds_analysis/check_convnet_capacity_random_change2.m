function check_convnet_capacity_random_change2(P, range_factor, N_SAMPLES, network_type, degrees_of_freedom, local_preprocessing, input_suffix, run_id, random_labeling_type, layers_grouping_level, samples_jump, features_type, N_OBJECTS)
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
    if network_type >= 2 && network_type <= 4
        layers_grouping_level = 2;
    else
        layers_grouping_level = 0;
    end
end
if nargin < 11
    samples_jump = 1;
end
if nargin < 12
    features_type = 2; % 0: sub-sample, 1: use the first n features (e.g. PCA), 2: random projections
end
if nargin < 13
    N_OBJECTS = P;
else
    assert(N_OBJECTS <= P);
end

global_preprocessing = 0; % 0- none, 1- znorm, 2- whitening, 3- centers decorelation

% Expected precision ep=sqrt(pq/n) so that for p=q=0.5 ep=0.05 yields n=100
EXPECTED_PRECISION = 0.05; 
max_samples = 100;
precision = 1;

i = strfind(input_suffix, 'seed');
if isempty(i)
    seed = 0;
else
    seed = sscanf(input_suffix(i:end), 'seed%d');
end

[network_name, N_LAYERS, ~, layer_names, ~, ~, ~, ~, ~, ~, ~, layers, ~] = ...
    load_network_metadata(network_type, layers_grouping_level, nan, seed);
ENABLED_LAYERS = zeros(1, N_LAYERS); ENABLED_LAYERS(layers) = 1;

input_prefix = sprintf('%s/generate_%s_random_change_dof%d', network_name, network_name, degrees_of_freedom);
global IMAGENET_IMAGE_SIZE;
if IMAGENET_IMAGE_SIZE ~= 64
    input_prefix = sprintf('%s_%dpx', input_prefix, IMAGENET_IMAGE_SIZE);
end

N_DIRECTIONS  = 2;
global N_HMAX_FEATURES;
N_NEURONS = N_HMAX_FEATURES;
assert(~isempty(N_NEURONS), 'Run init_imagenet');

function out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, global_preprocessing, local_preprocessing, degrees_of_freedom, input_suffix, samples_jump, features_type, run_id)
    prefix = sprintf('check_%s_capacity_random_change_dof%d', network_name, degrees_of_freedom);
    if IMAGENET_IMAGE_SIZE ~= 64
        prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
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
    if samples_jump ~= 1
        suffix = sprintf('%s_s%d', suffix, samples_jump);
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
neuron_samples_used_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS);
n_support_vectors_results   = nan(N_DIRECTIONS, N_LAYERS, N_NEURONS, N_OBJECTS);

if length(run_id) > 1
    run_ids = run_id;
    out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, global_preprocessing, local_preprocessing, degrees_of_freedom, input_suffix, samples_jump, features_type, 0);
    fprintf('Results saved to %s\n', out_name);
    missing = 0;
    for run_id=run_ids
        run_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, global_preprocessing, local_preprocessing, degrees_of_freedom, input_suffix, samples_jump, features_type, run_id);
        fprintf('Results loaded from %s\n', run_name);
        [l, param_id]=ind2sub([N_LAYERS, N_DIRECTIONS], run_id);
        assert(~ENABLED_LAYERS(l) || exist(run_name, 'file')>0);
        if ~exist(run_name, 'file')
            fprintf('Skipping missing file: %s\n', run_name);
	    missing = missing+ENABLED_LAYERS(l);
            continue;
        end
        run_file = matfile(run_name);
        % Fill results
        capacity_results(param_id, l) = run_file.capacity_results(param_id, l);
        separability_results(param_id, l, :) = run_file.separability_results(param_id, l, :);        
        neuron_samples_used_results(param_id, l, :) = run_file.neuron_samples_used_results(param_id, l, :);
        n_support_vectors_results(param_id, l, :, :) = run_file.n_support_vectors_results(param_id, l, :, :);
    end
    if missing>0
        fprintf('Missing %d results\n', missing);
    end
    % Save results
    save(out_name, 'capacity_results', 'separability_results', 'neuron_samples_used_results', 'n_support_vectors_results', ...
        'EXPECTED_PRECISION', 'max_samples', 'precision', 'features_type', '-v7.3');
    return;
end
        
out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, network_name, random_labeling_type, global_preprocessing, local_preprocessing, degrees_of_freedom, input_suffix, samples_jump, features_type, run_id);
fprintf('Results saved to %s\n', out_name);

if exist(out_name, 'file')
    fprintf('Loading existing results\n');
    load(out_name);
end

if run_id == 0
    RUN_DIRECTIONS = 1:N_DIRECTIONS;
    RUN_LAYERS = 1:N_LAYERS;
else
    fprintf('Run ID %d/%d\n', run_id, N_LAYERS*N_DIRECTIONS);
    [l, param_id]=ind2sub([N_LAYERS, N_DIRECTIONS], run_id);
    RUN_DIRECTIONS = param_id;
    RUN_LAYERS = l;
end

T=tic;
for param_id=RUN_DIRECTIONS
    for l=RUN_LAYERS
        if ~ENABLED_LAYERS(l)
            fprintf('Skipping disabled layer %s\n', layer_names{l});
            continue;
        end 

        %a = mean_argmax_norm2_results(param_id, l, :, :);
        %if isfinite(capacity_results(param_id, l)) && all(isfinite(a(:)))
        if isfinite(capacity_results(param_id, l))
            fprintf('Skipping existing %s\n', layer_names{l});
            continue;
        end
        fprintf('Working on %s\n', layer_names{l});

        if range_factor < 0.1
            in_name = sprintf('%s_range%f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_names{l}, input_suffix);
        else
            in_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_names{l}, input_suffix);
        end
        fprintf('Reading tuning function from %s\n', in_name);
        in_file = matfile(in_name);

        tic;
        full_tuning_function = double(permute(squeeze(in_file.tuning_function(param_id, 1:N_OBJECTS, :, :)), [3, 2, 1]));
        assert(all(size(full_tuning_function) == [N_NEURONS, N_SAMPLES, N_OBJECTS]));
        mean_squeare_firing_rate = mean(mean(full_tuning_function.^2, 3),2);
        nzIndices = find(mean_squeare_firing_rate > 0);
        N = length(nzIndices);
        fprintf('Loaded data (took %1.1f sec)\n', toc);
        
        if local_preprocessing == 0
            current_tuning_function = full_tuning_function(nzIndices,:,:);
        else
            minN = min(N_SAMPLES, N);
            current_tuning_function = calc_low_dimension_manifold(full_tuning_function(nzIndices,:,:), minN, local_preprocessing);
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
                precision, max_samples, global_preprocessing, features_type, min(512, N));
        
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
