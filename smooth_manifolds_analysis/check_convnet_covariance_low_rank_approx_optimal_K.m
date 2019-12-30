function check_convnet_covariance_low_rank_approx_optimal_K(P, range_factor, N_SAMPLES, network_type, run_id, properties_method, degrees_of_freedom, epoch, layers_grouping_level, ...
    samples_jump, N_OBJECTS, obj_id, n_neurons, seed, D, center_norm_factor, objects_seed)
if nargin < 5
    run_id = 0;
end
if nargin < 6
    % 0: norm scaling + legacy iterative method; 1: norm scaling + new LS method 2: Faster, norm scaling + w/o iterative values; 
    % 4: nullspace method + legacy iterative method 5: nullspace method + LS method
    properties_method = 0; 
end
if nargin < 7
    degrees_of_freedom = 1; % 1:1D, 2:2D
end
if nargin < 8
    epoch = nan;
end
if nargin < 9
    if network_type >= 2 && network_type <= 4
        layers_grouping_level = 2;
    else
        layers_grouping_level = 0;
    end
end
if nargin < 10
    samples_jump = 1;
end
if nargin < 11
    N_OBJECTS = P;
else
    assert(N_OBJECTS <= P);
end
if nargin < 12
    obj_id = 0;
end
if nargin < 13
    n_neurons = 0;
end
if nargin < 14
    seed = 0;
end
if nargin < 15
    D = nan;
end
if nargin < 16
    center_norm_factor = 1;
end
if nargin < 17
    objects_seed = 0;
end
minSquare = false;
optimization_verbose = 1;
use_method = bitand(properties_method, 1);
if use_method == 0
    use_method = bitand(properties_method, 2);
end
input_suffix = '';
if ~isnan(epoch)
    input_suffix = sprintf('%s_epoch%d', input_suffix, epoch);
end
if seed ~= 0
    input_suffix = sprintf('%s_seed%d', input_suffix, seed);
end
if objects_seed ~= 0
    input_suffix = sprintf('%s_objectsSeed%d', input_suffix, objects_seed);
end

% Network metadata
[network_name, N_LAYERS, ~, layer_names, ~, ~, ~, ~, ~, ~, ~, layers] = ...
    load_network_metadata(network_type, layers_grouping_level, epoch, seed);
ENABLED_LAYERS = zeros(1, N_LAYERS); ENABLED_LAYERS(layers) = 1;

if degrees_of_freedom == 1
    input_prefix = sprintf('%s/generate_%s_one_dimensional_change', network_name, network_name);
else
    input_prefix = sprintf('%s/generate_%s_random_change_dof%d', network_name, network_name, degrees_of_freedom);
end


global IMAGENET_IMAGE_SIZE;
if IMAGENET_IMAGE_SIZE ~= 64
    input_prefix = sprintf('%s_%dpx', input_prefix, IMAGENET_IMAGE_SIZE);
end

% Directions
if degrees_of_freedom == 1
    direction_names = {'x-translation', 'y-translation', 'x-scale', 'y-scale', 'x-shear', 'y-shear', 'rotation'};
elseif degrees_of_freedom == 2
    direction_names = {'x-y-translation', 'x-y-shear'};
else
    assert(degrees_of_freedom == 4);
    direction_names = {'translation and shear'};
end
N_DIRECTIONS = length(direction_names);
if n_neurons == 0
    N_NEURON_SAMPLES = 1;
else
    N_NEURON_SAMPLES = 10;
end
global N_HMAX_FEATURES;
MAX_NEURONS = N_HMAX_FEATURES;

MAX_K = min(ceil(N_OBJECTS/2),46);
OPTIMIZATION_N_REPEATS = 10;
AllKs = 0:MAX_K;
N_Ks=length(AllKs); 
%random_labeling_type = 1; % 0- binary iid, 1- balanced, 2- sparse

function [out_name] = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, obj_run_id, n_neurons, input_suffix, D, center_norm_factor)
    if degrees_of_freedom == 1
        prefix = sprintf('check_%s_covariance_low_rank_approx_optimal_K', network_name);
    else
        prefix = sprintf('check_%s_covariance_low_rank_approx_optimal_K_dof%d', network_name, degrees_of_freedom);
    end
    if IMAGENET_IMAGE_SIZE ~= 64
        prefix = sprintf('%s_%dpx', prefix, IMAGENET_IMAGE_SIZE);
    end
    if properties_method == 0
        output_suffix = '';
    elseif properties_method == 1
        output_suffix = '_slow';
    elseif properties_method == 2
        output_suffix = '_fast';
    elseif properties_method == 4
        output_suffix = '_nullspace';
    elseif properties_method == 5
        output_suffix = '_nullspace_slow';
    elseif properties_method == 6
        output_suffix = '_nullspace_Dplus1';
    else
        assert(false, 'Now supported yet');
    end
    
    suffix = '';
    if samples_jump ~= 1
        suffix = sprintf('%s_s%d', suffix, samples_jump);
    end
    if run_id > 0
        suffix = sprintf('%s_%d', suffix, run_id);
    end
    if obj_run_id > 0
        suffix = sprintf('%s_o%d', suffix, obj_run_id);
    end
    if n_neurons>0
        suffix = sprintf('%s_n%d', suffix, n_neurons);
    end
    if ~isnan(D)
        suffix = sprintf('%s_D%d', suffix, D);
    end
    if center_norm_factor ~= 1
        suffix = sprintf('%s_f%1.2f', suffix, center_norm_factor);
    end

    if range_factor < 0.1
        out_name = sprintf('%s_range%f_P%d_M%d%s%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, suffix, output_suffix, input_suffix);
    else
        out_name = sprintf('%s_range%1.1f_P%d_M%d%s%s%s.mat', prefix, range_factor, N_OBJECTS, N_SAMPLES, suffix, output_suffix, input_suffix);
    end
end

% Results variables
best_K_results = nan(N_DIRECTIONS, N_LAYERS);
effective_n_results = nan(N_DIRECTIONS, N_LAYERS);

correlation_spectrum_results = nan(N_DIRECTIONS, N_LAYERS, N_OBJECTS);
normalized_correlation_spectrum_results = nan(N_DIRECTIONS, N_LAYERS, N_OBJECTS);

centers_norm_results = nan(N_DIRECTIONS, N_LAYERS, N_Ks, N_OBJECTS);
common_subspace_results = nan(N_DIRECTIONS, N_LAYERS, MAX_NEURONS, MAX_K);
residual_centers_results = nan(N_DIRECTIONS, N_LAYERS, MAX_NEURONS, N_OBJECTS);
residual_correlations_results = nan(N_DIRECTIONS, N_LAYERS, N_OBJECTS, N_OBJECTS);
original_correlations_results = nan(N_DIRECTIONS, N_LAYERS, N_OBJECTS, N_OBJECTS);

mean_square_corrcoef_results = nan(N_DIRECTIONS, N_LAYERS, N_Ks);
mean_abs_corrcoef_results = nan(N_DIRECTIONS, N_LAYERS, N_Ks);
mean_square_correlations_results = nan(N_DIRECTIONS, N_LAYERS, N_Ks);
mean_abs_correlations_results = nan(N_DIRECTIONS, N_LAYERS, N_Ks);

spectral_dimension_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
spectral_radius_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);

theory_capacity_results = nan(N_DIRECTIONS, N_LAYERS);
mean_half_width_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
mean_argmax_norm_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
mean_half_width2_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
mean_argmax_norm2_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
effective_dimension_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
effective_dimension2_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);
alphac_hat_results = nan(N_DIRECTIONS, N_LAYERS, N_NEURON_SAMPLES, N_OBJECTS);

out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, 0, obj_id, n_neurons, input_suffix, D, center_norm_factor);
if exist(out_name, 'file')
    fprintf('Loading previously collected results\n');    
    load(out_name);
end
if obj_id > 0
    out_name_id = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, 0, n_neurons, input_suffix, D, center_norm_factor);
    if exist(out_name_id, 'file')
        fprintf('Loading previously collected results from run %d\n', run_id);    
        load(out_name_id);
    end
end
    

if length(run_id) > 1
    assert(obj_id == 0, 'Must collect objects before collecting runs');
    run_ids = run_id;
    fprintf('Results saved to %s\n', out_name);
    missing = 0;
    for run_id=run_ids
        run_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, obj_id, n_neurons, input_suffix, D, center_norm_factor);
    	fprintf('Loading %s\n', run_name);
        [l, param_id] = ind2sub([N_LAYERS, N_DIRECTIONS], run_id);
        assert(~ENABLED_LAYERS(l) || exist(run_name, 'file')>0);
        if ~exist(run_name, 'file')
            fprintf('Skipping missing file...\n');
            missing = missing + ENABLED_LAYERS(l);
            continue;
        end
        run_file = matfile(run_name);
        best_K_results(param_id, l) = run_file.best_K_results(param_id, l);
        if isprop(run_file, 'effective_n_results')
            effective_n_results(param_id, l) = run_file.effective_n_results(param_id, l);
        end
        L = length(run_file.AllKs);
        assert(all(AllKs(1:L) == run_file.AllKs), 'Not the same number of K-s'); 
        J = 1:size(run_file, 'common_subspace_results', 4);
        common_subspace_results(param_id, l, :, J) = run_file.common_subspace_results(param_id, l, :, :);

        I = 1:size(run_file, 'centers_norm_results', 3);
        centers_norm_results(param_id, l, I, :) = run_file.centers_norm_results(param_id, l, :, :);
        mean_square_corrcoef_results(param_id, l, I) = run_file.mean_square_corrcoef_results(param_id, l, :);
        mean_abs_corrcoef_results(param_id, l, I) = run_file.mean_abs_corrcoef_results(param_id, l, :);
        mean_square_correlations_results(param_id, l, I) = run_file.mean_square_correlations_results(param_id, l, :);
        mean_abs_correlations_results(param_id, l, I) = run_file.mean_abs_correlations_results(param_id, l, :);
        
        correlation_spectrum_results(param_id, l, :) = run_file.correlation_spectrum_results(param_id, l, :);
        normalized_correlation_spectrum_results(param_id, l, :) = run_file.normalized_correlation_spectrum_results(param_id, l, :);
        residual_correlations_results(param_id, l, :, :) = run_file.residual_correlations_results(param_id, l, :, :);
        if isprop(run_file, 'original_correlations_results')
            original_correlations_results(param_id, l, :, :) = run_file.original_correlations_results(param_id, l, :, :);
        end
    	residual_centers_results(param_id, l, :, :) = run_file.residual_centers_results(param_id, l, :, :);

        a = run_file.spectral_dimension_results(param_id, l, :, :);
        assert_warn(all(isfinite(a(:))), 'Code did not finish executing!');
        %if all(isfinite(a(:)))
            spectral_dimension_results(param_id, l, :, :) = run_file.spectral_dimension_results(param_id, l, :, :);
            spectral_radius_results(param_id, l, :, :) = run_file.spectral_radius_results(param_id, l, :, :);
        %end
        if properties_method == 2
            a = run_file.mean_half_width_results(param_id, l, :, :);
        else
            a = run_file.theory_capacity_results(param_id, l);
        end
        assert_warn(all(isfinite(a(:))), 'Code did not finish executing!');
        %if all(isfinite(a(:)))
        theory_capacity_results(param_id, l) = run_file.theory_capacity_results(param_id, l);
        mean_half_width_results(param_id, l, :, :) = run_file.mean_half_width_results(param_id, l, :, :);
        mean_argmax_norm_results(param_id, l, :, :) = run_file.mean_argmax_norm_results(param_id, l, :, :);
        mean_half_width2_results(param_id, l, :, :) = run_file.mean_half_width2_results(param_id, l, :, :);
        mean_argmax_norm2_results(param_id, l, :, :) = run_file.mean_argmax_norm2_results(param_id, l, :, :);
        effective_dimension_results(param_id, l, :, :) = run_file.effective_dimension_results(param_id, l, :, :);
        effective_dimension2_results(param_id, l, :, :) = run_file.effective_dimension2_results(param_id, l, :, :);
        alphac_hat_results(param_id, l, :, :) = run_file.alphac_hat_results(param_id, l, :, :);
        %end
    end
    save(out_name, 'OPTIMIZATION_N_REPEATS', 'AllKs', 'best_K_results', 'effective_n_results', ...
        'theory_capacity_results', 'residual_centers_results', 'centers_norm_results', ...
        'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
        'mean_square_correlations_results', 'mean_abs_correlations_results', ...
        'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
        'original_correlations_results', 'residual_correlations_results', ...
        'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
        'mean_half_width_results', 'mean_argmax_norm_results', ...
        'mean_half_width2_results', 'mean_argmax_norm2_results', ...
        'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
    missing2 = isnan(theory_capacity_results(:,ENABLED_LAYERS==1)); missing2=sum(missing2(:));
    total = numel(theory_capacity_results);
    fprintf('%d values (%d files, %1.1f%%) are still missing\n', missing2, missing, 100.0*missing2/total);
    return;
end
if length(obj_id) > 1
    out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, 0, n_neurons, input_suffix, D, center_norm_factor);
    if exist(out_name, 'file')
        fprintf('Loading previously collected results for run %d\n', run_id);
        load(out_name);
    end

    assert(run_id > 0, 'Must collect objects for specific run');
    obj_ids = obj_id;
    fprintf('Results saved to %s\n', out_name);
    missing = 0;
    [l, param_id] = ind2sub([N_LAYERS, N_DIRECTIONS], run_id);
    for obj_id=obj_ids
        run_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, obj_id, n_neurons, input_suffix, D, center_norm_factor);
    	fprintf('Loading %s\n', run_name);
        assert(~ENABLED_LAYERS(l) || exist(run_name, 'file')>0, run_name);
        if ~exist(run_name, 'file')
            fprintf('Skipping missing file...\n');
            missing = missing + ENABLED_LAYERS(l);
            continue;
        end
        run_file = matfile(run_name);
        if isfinite(best_K_results(param_id, l))
            assert(best_K_results(param_id, l) == run_file.best_K_results(param_id, l));
        else
            best_K_results(param_id, l) = run_file.best_K_results(param_id, l);
        end
        if isprop(run_file, 'effective_n_results')
            if isfinite(effective_n_results(param_id, l))
                assert(effective_n_results(param_id, l) == run_file.effective_n_results(param_id, l));
            else
                effective_n_results(param_id, l) = run_file.effective_n_results(param_id, l);
            end
        end
        L = length(run_file.AllKs);
        assert(all(AllKs(1:L) == run_file.AllKs), 'Not the same number of K-s'); 
        J = 1:size(run_file, 'common_subspace_results', 4);
        if all(isfinite(common_subspace_results(param_id, l, :, J)))
            assert(norm(common_subspace_results(param_id, l, :, J) - run_file.common_subspace_results(param_id, l, :, :)) < 1e-10);
        else
            common_subspace_results(param_id, l, :, J) = run_file.common_subspace_results(param_id, l, :, :);
        end

        a = centers_norm_results(param_id, l, :, :);
        if all(~isfinite(a(:)))
            I = 1:size(run_file, 'centers_norm_results', 3);
            centers_norm_results(param_id, l, I, :) = run_file.centers_norm_results(param_id, l, :, :);
            mean_square_corrcoef_results(param_id, l, I) = run_file.mean_square_corrcoef_results(param_id, l, :);
            mean_abs_corrcoef_results(param_id, l, I) = run_file.mean_abs_corrcoef_results(param_id, l, :);
            mean_square_correlations_results(param_id, l, I) = run_file.mean_square_correlations_results(param_id, l, :);
            mean_abs_correlations_results(param_id, l, I) = run_file.mean_abs_correlations_results(param_id, l, :);

            correlation_spectrum_results(param_id, l, :) = run_file.correlation_spectrum_results(param_id, l, :);
            normalized_correlation_spectrum_results(param_id, l, :) = run_file.normalized_correlation_spectrum_results(param_id, l, :);
            residual_correlations_results(param_id, l, :, :) = run_file.residual_correlations_results(param_id, l, :, :);
            if isprop(run_file, 'original_correlations_results')
                original_correlations_results(param_id, l, :, :) = run_file.original_correlations_results(param_id, l, :, :);
            end
            residual_centers_results(param_id, l, :, :) = run_file.residual_centers_results(param_id, l, :, :);
        end
        a = run_file.spectral_dimension_results(param_id, l, :, obj_id);
        assert(all(isfinite(a(:))), 'Code did not finish executing!');
        spectral_dimension_results(param_id, l, :, obj_id) = run_file.spectral_dimension_results(param_id, l, :, obj_id);
        spectral_radius_results(param_id, l, :, obj_id) = run_file.spectral_radius_results(param_id, l, :, obj_id);
        
        a = run_file.mean_half_width_results(param_id, l, :, obj_id);
        assert(all(isfinite(a(:))), 'Code did not finish executing!');
        mean_half_width_results(param_id, l, :, obj_id) = run_file.mean_half_width_results(param_id, l, :, obj_id);
        mean_argmax_norm_results(param_id, l, :, obj_id) = run_file.mean_argmax_norm_results(param_id, l, :, obj_id);
        mean_half_width2_results(param_id, l, :, obj_id) = run_file.mean_half_width2_results(param_id, l, :, obj_id);
        mean_argmax_norm2_results(param_id, l, :, obj_id) = run_file.mean_argmax_norm2_results(param_id, l, :, obj_id);
        effective_dimension_results(param_id, l, :, obj_id) = run_file.effective_dimension_results(param_id, l, :, obj_id);
        effective_dimension2_results(param_id, l, :, obj_id) = run_file.effective_dimension2_results(param_id, l, :, obj_id);
        alphac_hat_results(param_id, l, :, obj_id) = run_file.alphac_hat_results(param_id, l, :, obj_id);
    end
    alphac_hat = squeeze(alphac_hat_results(param_id, l, :, :));
    Ac = mean(1./mean(1./alphac_hat,2),1);
    assert(all(Ac), 'Code did not finish executing!');
    theory_capacity_results(param_id, l) = Ac;
    save(out_name, 'OPTIMIZATION_N_REPEATS', 'AllKs', 'best_K_results', 'effective_n_results', ...
        'theory_capacity_results', 'residual_centers_results', 'centers_norm_results', ...
        'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
        'mean_square_correlations_results', 'mean_abs_correlations_results', ...
        'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
        'original_correlations_results', 'residual_correlations_results', ...
        'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
        'mean_half_width_results', 'mean_argmax_norm_results', ...
        'mean_half_width2_results', 'mean_argmax_norm2_results', ...
        'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
    fprintf('%d files are still missing\n', missing);
    return;
end

out_name = get_filenames(range_factor, N_OBJECTS, N_SAMPLES, properties_method, degrees_of_freedom, samples_jump, run_id, obj_id, n_neurons, input_suffix, D, center_norm_factor);
fprintf('Results saved to %s (@%s)\n', out_name, strtrim(hostname()));
assert(obj_id == 0 || exist(out_name, 'file')>0, 'Cannot run specific object when the common subspace was not calculated');

if run_id == 0
    DIRECTIONS = 1:N_DIRECTIONS;
    LAYERS = 1:N_LAYERS;
else
    [l, param_id] = ind2sub([N_LAYERS, N_DIRECTIONS], run_id);
    DIRECTIONS = param_id;
    LAYERS = l;
end

if exist(out_name, 'file')
    fprintf('Loading previous results\n');
    load(out_name);
end

for param_id=DIRECTIONS
    for l=LAYERS 
        if ~ENABLED_LAYERS(l)
            fprintf('Skipping disabled layer %s\n', layer_names{l});
            continue;
        end
        % The objects of interest in this run
        if obj_id == 0
            objI = 1:N_OBJECTS;
        else
            objI = obj_id;
        end
                
        a1 = spectral_dimension_results(param_id, l, :, objI);
        a2 = mean_half_width_results(param_id, l, :, objI);
        b = centers_norm_results(param_id,l,1,:);
        valid_values = all(b(isfinite(b(:))) < 1e5);
        if ~valid_values; fprintf('Found invalid value (%1.3e)\n', max(b(:))); end
        if valid_values && isfinite(best_K_results(param_id, l)) && all(isfinite(a1(:))) && all(isfinite(a2(:)))
            fprintf('Skipping existing results on %s %s\n', direction_names{param_id}, layer_names{l});
            continue;
        end
        fprintf('Working on %s %s\n', direction_names{param_id}, layer_names{l});

        if range_factor < 0.1
            in_name = sprintf('%s_range%f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_names{l}, input_suffix);
        else
            in_name = sprintf('%s_range%1.1f_P%d_M%d_%s%s.mat', input_prefix, range_factor, P, N_SAMPLES, layer_names{l}, input_suffix);
        end
        fprintf('Reading tuning function from %s\n', in_name);
        in_file = matfile(in_name);

        tic;
        full_tuning_function = double(permute(squeeze(in_file.tuning_function(param_id, 1:N_OBJECTS, :, :)), [3, 2, 1]));
        N_NEURONS = size(full_tuning_function, 1);
        assert(all(size(full_tuning_function) == [N_NEURONS, N_SAMPLES, N_OBJECTS]));
        mean_squeare_firing_rate = mean(mean(full_tuning_function.^2, 3),2);
        nzIndices = find(mean_squeare_firing_rate > 0);
        N = length(nzIndices);
        effective_n_results(param_id, l) = N;
        fprintf('Loaded data (took %1.1f sec)\n', toc);
        
        % Remove all zero neurons
        current_tuning_function = full_tuning_function(nzIndices, :, :);

        if samples_jump == 0 
            % Use a single sample
            assert(mod(N_SAMPLES,2) == 1, 'N_SAMPLES must be odd');
            current_tuning_function = current_tuning_function(:,(N_SAMPLES+1)/2,:);
        else
            assert(mod(N_SAMPLES-1, samples_jump) == 0, 'samples_jump must divide N_SAMPLES-1');
            current_tuning_function = current_tuning_function(:,1:samples_jump:N_SAMPLES,:);
            N_SAMPLES = size(current_tuning_function,2);
        end

        % Substract the global mean
        current_tuning_function = bsxfun(@minus, current_tuning_function, mean(mean(current_tuning_function, 2),3));

        % Calculate centers
        centers = reshape(mean(current_tuning_function, 2), [N, N_OBJECTS]);
        centers_norm = sqrt(sum(centers.^2,1));

        % Non-normalized covariance matrix
        C = centers'*centers;
        correlation_spectrum_results(param_id, l, :) = svd(C);

        % Normalized covariance matrix
        C0 = C ./ (centers_norm'*centers_norm);
        assert(max(abs(diag(C0)-1))<1e-10);
        original_correlations_results(param_id, l, :, :) = C0;
        normalized_correlation_spectrum_results(param_id, l, :) = svd(C0);
        
        if valid_values && isfinite(best_K_results(param_id, l))
            fprintf('Skipping existing results for common sub-space...\n');
            Kopt = best_K_results(param_id, l);
            Xk = reshape(residual_centers_results(param_id, l, 1:N, :), [N, N_OBJECTS]);
            Vk = reshape(common_subspace_results(param_id, l, 1:N, 1:Kopt), [N, Kopt]);
        else
            [Vk, Xk, Kopt, residual_centers_norm, mean_square_corrcoef, mean_abs_corrcoef, mean_square_corr, mean_abs_corr] = ...
                optimal_low_rank_structure2(centers, MAX_K, optimization_verbose, minSquare, OPTIMIZATION_N_REPEATS);
            best_K_results(param_id, l) = Kopt;
            centers_norm_results(param_id, l, :, :) = residual_centers_norm;
            mean_abs_correlations_results(param_id, l, :) = mean_abs_corr;
            mean_square_correlations_results(param_id, l, :) = mean_square_corr;
            mean_square_corrcoef_results(param_id, l, :) = mean_square_corrcoef;
            mean_abs_corrcoef_results(param_id, l, :) = mean_abs_corrcoef;
            residual_centers_results(param_id, l, 1:N, :) = Xk;
            common_subspace_results(param_id, l, 1:N, 1:Kopt) = Vk;
            residual_centers_norm = sqrt(sum(Xk.^2,1));
            residual_correlations_results(param_id, l, :, :) = Xk'*Xk./(residual_centers_norm'*residual_centers_norm);

            fprintf('Saving results...\n');
            save(out_name, 'OPTIMIZATION_N_REPEATS', 'AllKs', 'best_K_results', 'effective_n_results', ...
                'theory_capacity_results', 'residual_centers_results', 'centers_norm_results', ...
                'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
                'mean_square_correlations_results', 'mean_abs_correlations_results', ...
                'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
                'original_correlations_results', 'residual_correlations_results', ...
                'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
                'mean_half_width_results', 'mean_argmax_norm_results', ...
                'mean_half_width2_results', 'mean_argmax_norm2_results', ...
                'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
        end
        %if obj_id == 0
	%   fprintf('Done global processing, run specific objects\n');
	%   continue;
	%end

        % Calculate tuning in the null-space of the common space
        common_components = reshape(Vk*(Vk'*reshape(current_tuning_function, [N, N_SAMPLES*N_OBJECTS])), size(current_tuning_function));
        projected_tuning_function = current_tuning_function - common_components;
        projected_centers = squeeze(mean(projected_tuning_function, 2));
        if bitand(properties_method, 4) == 4
            clear current_tuning_function;
        end

        if n_neurons == 0
            features_used = 1:N; % Calculate properties once using all features
            n_neurons = N;
        else
            assert(n_neurons <= N, 'Not enough neurons to sample from');
            features_used = sample_indices(N, n_neurons, N_NEURON_SAMPLES);
        end
        assert(all(size(features_used) == [N_NEURON_SAMPLES, n_neurons]));
        
        for ir=1:N_NEURON_SAMPLES
            a = spectral_dimension_results(param_id, l, ir, objI);
            if valid_values && all(isfinite(a(:)))
                fprintf('Skipping existing results for spectral properties...\n');
            else
                tic;
                minN = min(n_neurons, N_SAMPLES);
                for j=objI
                    cF = squeeze(bsxfun(@minus, projected_tuning_function(features_used(ir,:),:,j), projected_centers(features_used(ir,:),j)));
                    [~, S, ~] = svd(cF, 0);
		    d=D; if isnan(D); d=minN; end
                    SD = zeros(size(S)); SD(1:d,1:d) = S(1:d,1:d);
                    s = diag(SD);
                    
                    assert(length(s) == minN);
                    sq = s.^2/N_SAMPLES;
                    spectral_dimension_results(param_id, l, ir, j) = sum(sq).^2./sum(sq.^2);
                    spectral_radius_results(param_id, l, ir, j) = sqrt(sum(sq.^2)) ./ norm(center_norm_factor*projected_centers(:,j)).^2;
                end
                fprintf('%d spectral R=%1.3f D=%1.3f (took %1.1f sec)\n', ir, ...
                    mean(mean(spectral_radius_results(param_id, l, ir, objI),3),4), ...
                    mean(mean(spectral_dimension_results(param_id, l, ir, objI),3),4), toc);
            end
            if properties_method == 2
                a = mean_half_width_results(param_id, l, ir, objI);
            else
                a = mean_half_width2_results(param_id, l, ir, objI);
            end
            if valid_values && all(isfinite(a(:)))
                fprintf('%d skipping existing results for manifold properties...\n', ir);
                if properties_method == 2
                    mean_half_width1 = mean_half_width_results(param_id, l, ir, objI);
                    mean_argmax_norm1 = mean_argmax_norm_results(param_id, l, ir, objI);
                    gaussian_dimension = mean(mean_half_width1(:).^2./mean_argmax_norm1(:));
                    gaussian_radius = mean(sqrt(mean_argmax_norm1(:)));
                    fprintf('%d gaussian R=%1.3f D=%1.3f\n', ir, gaussian_radius, gaussian_dimension);
                else
                    mean_half_width2 = mean_half_width2_results(param_id, l, ir, objI);
                    mean_argmax_norm2 = mean_argmax_norm2_results(param_id, l, ir, objI);
                    manifold_dimension = mean(mean_half_width2(:).^2./mean_argmax_norm2(:));
                    manifold_radius = mean(sqrt(mean_argmax_norm2(:)));
                    fprintf('%d manifold R=%1.3f D=%1.3f\n', ir, manifold_radius, manifold_dimension);
                end
            else
                tic;
                if properties_method == 2
                    [mean_half_width1, mean_argmax_norm1] = ...
                        check_hierarchial_capacity(current_tuning_function, features_used(ir,:), centers, Xk, use_method, [], false, objI, D, center_norm_factor);
                    mean_half_width_results(param_id, l, ir, objI) = mean_half_width1(:, objI);
                    mean_argmax_norm_results(param_id, l, ir, objI) = mean_argmax_norm1(:, objI);
                    gaussian_dimension = mean(mean_half_width1(:).^2./mean_argmax_norm1(:));
                    gaussian_radius = mean(sqrt(mean_argmax_norm1(:)));
                    fprintf('%d gaussian R=%1.3f D=%1.3f (took %1.1f sec)\n', ir, gaussian_radius, gaussian_dimension, toc);
                elseif bitand(properties_method, 4) == 4
                    [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
                        check_samples_capacity(projected_tuning_function, features_used(ir,:), projected_centers, use_method, [], false, objI, D, center_norm_factor);
                    mean_half_width_results(param_id, l, ir, objI) = mean_half_width1(:, objI);
                    mean_argmax_norm_results(param_id, l, ir, objI) = mean_argmax_norm1(:, objI);
                    mean_half_width2_results(param_id, l, ir, objI) = mean_half_width2(:, objI);
                    mean_argmax_norm2_results(param_id, l, ir, objI) = mean_argmax_norm2(:, objI);
                    effective_dimension_results(param_id, l, ir, objI) = effective_dimension(:, objI);
                    effective_dimension2_results(param_id, l, ir, objI) = effective_dimension2(:, objI);
                    alphac_hat_results(param_id, l, ir, objI) = alphac_hat(:, objI);
                    %alphac_hat2=theory_alpha0_cached(mean_half_width2).*(1+mean_argmax_norm2);
                    %Ac = mean(1./mean(1./alphac_hat2,2),1);
                    if obj_id == 0
                        Ac = mean(1./mean(1./squeeze(alphac_hat_results(param_id, l, :, :)),2),1);
                        theory_capacity_results(param_id, l) = Ac;
                    else
                        Ac = nan;
                    end
                    fprintf('%d k=%d: Ac=%1.3f W=%1.3f D=%1.3f (took %1.1f sec)\n', ir, Kopt, Ac, mean(mean_half_width2,2), mean(effective_dimension2,2), toc);
                else
                    [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
                        check_hierarchial_capacity(current_tuning_function, features_used(ir,:), centers, Xk, use_method, [], false, objI, D, center_norm_factor);
                    mean_half_width_results(param_id, l, ir, objI) = mean_half_width1(:, objI);
                    mean_argmax_norm_results(param_id, l, ir, objI) = mean_argmax_norm1(:, objI);
                    mean_half_width2_results(param_id, l, ir, objI) = mean_half_width2(:, objI);
                    mean_argmax_norm2_results(param_id, l, ir, objI) = mean_argmax_norm2(:, objI);
                    effective_dimension_results(param_id, l, ir, objI) = effective_dimension(:, objI);
                    effective_dimension2_results(param_id, l, ir, objI) = effective_dimension2(:, objI);
                    alphac_hat_results(param_id, l, ir, objI) = alphac_hat(:, objI);
                    if objI == 0
                        Ac = mean(1./mean(1./squeeze(alphac_hat_results(param_id, l, :, :)),2),1);
                        theory_capacity_results(param_id, l) = Ac;
                    else
                        Ac = nan;
                    end
                    fprintf('%d k=%d: Ac=%1.3f W=%1.3f D=%1.3f (took %1.1f sec)\n', ir, Kopt, Ac, mean(mean_half_width2(:, objI),2), mean(effective_dimension2(:, objI),2), toc);
                end
                fprintf('Saving results...\n');
                save(out_name, 'OPTIMIZATION_N_REPEATS', 'AllKs', 'best_K_results', 'effective_n_results', ...
                'theory_capacity_results', 'residual_centers_results', 'centers_norm_results', ...
                    'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
                    'mean_square_correlations_results', 'mean_abs_correlations_results', ...
                    'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
                    'original_correlations_results', 'residual_correlations_results', ...
                    'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
                    'mean_half_width_results', 'mean_argmax_norm_results', ...
                    'mean_half_width2_results', 'mean_argmax_norm2_results', ...
                    'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
            end
        end
    end
end
end
