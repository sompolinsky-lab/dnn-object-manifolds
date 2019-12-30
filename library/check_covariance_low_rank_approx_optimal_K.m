function check_covariance_low_rank_approx_optimal_K(prefix, session_ids, n_neurons, properties_method, minSquare, data_randomization, seed, n_repeats, samples_jump)
if nargin < 2
    session_ids = 0;
end
if nargin < 3
    n_neurons = 0;
end
if nargin < 4
    properties_method = 5;
end
if nargin < 5
    minSquare = false;
end
if nargin < 6
    data_randomization = 0;
end
if nargin < 7
    seed = [];
end
if nargin < 8
    n_repeats = 10;
end
if nargin < 9
    samples_jump = 1;
end

optimization_verbose = 1;
use_method = bitand(properties_method, 1);
if use_method == 0
    use_method = bitand(properties_method, 2);
end

in_name = sprintf('%s_tuning.mat', prefix);
assert(exist(in_name, 'file')>0, 'File not found: %s', in_name);
in_file = matfile(in_name);

method = '';
if use_method == 0
    method = [method, '_legacy'];
elseif use_method == 2
    method = [method, '_Dplus1'];
end
if minSquare
    method = [method, '_minSqr'];
else
    method = [method, '_minAbs'];
end
if n_repeats>1
    method = sprintf('%s_r%d', method, n_repeats);
end
if ~isempty(seed)
    method = sprintf('%s_seed%d', method, seed);
end
if length(session_ids) == 1 && session_ids>0
    method = sprintf('%s_s%d', method, session_ids);
end
if n_neurons>0
    method = sprintf('%s_n%d', method, n_neurons);
end
if samples_jump ~= 1
    method = sprintf('%s_j%d', method, samples_jump);
end
if data_randomization == 1
    method = [method, '_shuffle'];
elseif data_randomization == 2
    method = [method, '_shuffle2'];
elseif data_randomization == 4
    method = [method, '_znorm'];
end
out_name = sprintf('%s_lowrank_optimalK%s.mat', prefix, method);

tuning_size = size(in_file, 'tuning_function');
if length(tuning_size) == 4
    [N_SESSIONS, N_NEURONS, N_SAMPLES, N_OBJECTS] = size(in_file, 'tuning_function');
else
    assert(length(tuning_size) == 3, 'Tuning function is not of size 3,4');
    N_SESSIONS = 1;
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(in_file, 'tuning_function');
end
assert(all(session_ids <= N_SESSIONS), 'session_id must be smaller than N_SESSIONS');
if N_NEURONS == 0
    fprintf('Warning: no neurons found\n');
    return;
end

% Get session titles
if isprop(in_file, 'data_titles')
    data_titles = in_file.data_titles;
else
    data_titles = cell(N_SESSIONS,1);
    for s=1:N_SESSIONS
        data_titles{s} = sprintf('session #%d', s);
    end
end
% Number of neuron samples
if n_neurons == 0
    N_NEURON_SAMPLES = 1;
else
    N_NEURON_SAMPLES = 10;
end
% Algorith parameters
MAX_K = min(ceil(N_OBJECTS/2),46);  % Components to remove
AllKs = 0:MAX_K;
N_Ks = length(AllKs);

% Results variables
best_K_results = nan(N_SESSIONS, 1);
effective_n_results = nan(N_SESSIONS, 1);

correlation_spectrum_results = nan(N_SESSIONS, N_OBJECTS);
normalized_correlation_spectrum_results = nan(N_SESSIONS, N_OBJECTS);

centers_norm_results = nan(N_SESSIONS, N_Ks, N_OBJECTS);
common_subspace_results = nan(N_SESSIONS, N_NEURONS, MAX_K);
residual_centers_results = nan(N_SESSIONS, N_NEURONS, N_OBJECTS);
residual_correlations_results = nan(N_SESSIONS, N_OBJECTS, N_OBJECTS);
original_correlations_results = nan(N_SESSIONS, N_OBJECTS, N_OBJECTS);

mean_square_corrcoef_results = nan(N_SESSIONS, N_Ks);
mean_abs_corrcoef_results = nan(N_SESSIONS, N_Ks);
mean_square_correlations_results = nan(N_SESSIONS, N_Ks);
mean_abs_correlations_results = nan(N_SESSIONS, N_Ks);

spectral_dimension_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
spectral_radius_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);

theory_capacity_results = nan(N_SESSIONS, 1);
mean_half_width_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
mean_argmax_norm_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
mean_half_width2_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
mean_argmax_norm2_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
effective_dimension_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
effective_dimension2_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);
alphac_hat_results = nan(N_SESSIONS, N_NEURON_SAMPLES, N_OBJECTS);

fprintf('Results saved to %s\n', out_name);
if exist(out_name, 'file')
    fprintf('Loading existing results\n');
    load(out_name);
end

if length(session_ids)>1
    fprintf('Collecting previousy generated results\n');
    for s=session_ids
        run_name = sprintf('%s_lowrank_optimalK%s_s%d.mat', prefix, method, s);
        assert(exist(run_name, 'file')>0, 'Missing file: %s', run_name);
        run_file = matfile(run_name);

        best_K_results(s,:) = run_file.best_K_results(s,:);
        if isprop(run_file, 'effective_n_results')
           effective_n_results(s,:) = run_file.effective_n_results(s,:);
        end
        correlation_spectrum_results(s,:) = run_file.correlation_spectrum_results(s,:);
        normalized_correlation_spectrum_results(s,:) = run_file.normalized_correlation_spectrum_results(s,:);

        centers_norm_results(s,:,:) = run_file.centers_norm_results(s,:,:);
        common_subspace_results(s,:,:) = run_file.common_subspace_results(s,:,:);
        residual_centers_results(s,:,:) = run_file.residual_centers_results(s,:,:);
        residual_correlations_results(s,:,:) = run_file.residual_correlations_results(s,:,:);
        original_correlations_results(s,:,:) = run_file.original_correlations_results(s,:,:);

        mean_square_corrcoef_results(s,:) = run_file.mean_square_corrcoef_results(s,:);
        mean_abs_corrcoef_results(s,:) = run_file.mean_abs_corrcoef_results(s,:);
        mean_square_correlations_results(s,:) = run_file.mean_square_correlations_results(s,:);
        mean_abs_correlations_results(s,:) = run_file.mean_abs_correlations_results(s,:);

        spectral_dimension_results(s,:,:) = run_file.spectral_dimension_results(s,:,:);
        spectral_radius_results(s,:,:) = run_file.spectral_radius_results(s,:,:);

        theory_capacity_results(s,:) = run_file.theory_capacity_results(s,:);
        mean_half_width_results(s,:,:) = run_file.mean_half_width_results(s,:,:);
        mean_argmax_norm_results(s,:,:) = run_file.mean_argmax_norm_results(s,:,:);
        mean_half_width2_results(s,:,:) = run_file.mean_half_width2_results(s,:,:);
        mean_argmax_norm2_results(s,:,:) = run_file.mean_argmax_norm2_results(s,:,:);
        effective_dimension_results(s,:,:) = run_file.effective_dimension_results(s,:,:);
        effective_dimension2_results(s,:,:) = run_file.effective_dimension2_results(s,:,:);
        alphac_hat_results(s,:,:) = run_file.alphac_hat_results(s,:,:);
    end
    assert_warn(all(isfinite(theory_capacity_results(session_ids))), 'Results did not finish running (%d)', sum(~isfinite(theory_capacity_results(session_ids))));
    
    save(out_name, 'AllKs', 'best_K_results', 'effective_n_results', 'theory_capacity_results', ...
        'residual_centers_results', 'centers_norm_results', ...
        'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
        'mean_square_correlations_results', 'mean_abs_correlations_results', ...
        'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
        'original_correlations_results', 'residual_correlations_results', ...
        'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
        'mean_half_width_results', 'mean_argmax_norm_results', ...
        'mean_half_width2_results', 'mean_argmax_norm2_results', ...
        'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
    return;
end

if session_ids == 0
    session_ids = 1:N_SESSIONS;
end

if ~isempty(seed)
    rng(seed);
end
T=tic;
for s=session_ids
    data_title = data_titles{s};
    a = spectral_dimension_results(s, :, :);
    b = mean_half_width_results(s, :, :);
    if all(isfinite(a(:))) && all(isfinite(b(:)))
        fprintf('Skipping existing session %s\n', data_title);
        continue;
    end
    fprintf('Working on session %s\n', data_title);

    tic;
    if length(tuning_size) == 3
        full_tuning_function = in_file.tuning_function;
    else
        full_tuning_function = squeeze(in_file.tuning_function(s,:,:,:));
    end
    %mean_squeare_sample_response = mean(mean(full_tuning_function.^2, 3),1); 
    %assert(length(mean_squeare_sample_response) == N_SAMPLES);
    %nzSamples = find(isfinite(mean_squeare_sample_response));
    %N_SAMPLES = length(nzSamples);
    %full_tuning_function = double(full_tuning_function(:,nzSamples,:));
    mean_squeare_firing_rate = nanmean(mean(full_tuning_function.^2, 3), 2);
    nzIndices = find(mean_squeare_firing_rate > 0);
    N = length(nzIndices);
    fprintf('Loaded %s data [N=%d M=%d P=%d] (took %1.1f sec)\n', data_title, N, N_SAMPLES, N_OBJECTS, toc);

    % Remove all zero neurons
    current_tuning_function = double(full_tuning_function(nzIndices, :, :));
    
    if data_randomization == 4
        neuron_mean = mean(current_tuning_function(:,:),2);
        neuron_std = std(current_tuning_function(:,:),[],2);
        current_tuning_function = (current_tuning_function - neuron_mean) ./ neuron_std;
    end

    % Use only some of the samples if needed
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
    current_tuning_function = bsxfun(@minus, current_tuning_function, mean(nanmean(current_tuning_function, 2),3));
    
    % Calculate properties once using all features
    effective_n_results(s) = N;
    if N < n_neurons 
        fprintf('Warning: Not enough neurons to sample from (%d)\n', N);
        continue;
    end

    % Calculate centers
    centers = reshape(nanmean(current_tuning_function, 2), [N, N_OBJECTS]);
    centers_norm = sqrt(sum(centers.^2,1));

    % Non-normalized covariance matrix
    C = centers'*centers;
    correlation_spectrum_results(s, :) = svd(C);

    % Normalized covariance matrix
    C0 = C ./ (centers_norm'*centers_norm);
    original_correlations_results(s, :, :) = C0;
    assert(max(abs(diag(C0)-1))<1e-4, 'Deviation from diagonal 1: %1.3e', max(abs(diag(C0)-1)));
    normalized_correlation_spectrum_results(s, :) = svd(C0);

    if isfinite(best_K_results(s))
        Kopt = best_K_results(s);
        Xk = reshape(residual_centers_results(s, 1:N, :), [N, N_OBJECTS]);
        Vk = reshape(common_subspace_results(s, 1:N, 1:Kopt), [N, Kopt]);
        fprintf('Skipping existing results for common sub-space, k=%d...\n', Kopt);
    else
        [Vk, Xk, Kopt, residual_centers_norm, mean_square_corrcoef, mean_abs_corrcoef, mean_square_corr, mean_abs_corr] = ...
            optimal_low_rank_structure2(centers, MAX_K, optimization_verbose, minSquare, n_repeats);
        best_K_results(s) = Kopt;
        centers_norm_results(s, :, :) = residual_centers_norm;
        mean_abs_correlations_results(s, :) = mean_abs_corr;
        mean_square_correlations_results(s, :) = mean_square_corr;
        mean_square_corrcoef_results(s, :) = mean_square_corrcoef;
        mean_abs_corrcoef_results(s, :) = mean_abs_corrcoef;
        residual_centers_results(s, 1:N, :) = Xk;
        common_subspace_results(s, 1:N, 1:Kopt) = Vk;
        residual_centers_norm = sqrt(sum(Xk.^2,1));
        residual_correlations_results(s, :, :) = Xk'*Xk./(residual_centers_norm'*residual_centers_norm);
        
        fprintf('Saving results...\n');
        save(out_name, 'AllKs', 'best_K_results', 'effective_n_results', 'theory_capacity_results', ...
            'residual_centers_results', 'centers_norm_results', ...
            'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
            'mean_square_correlations_results', 'mean_abs_correlations_results', ...
            'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
            'original_correlations_results', 'residual_correlations_results', ...
            'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
            'mean_half_width_results', 'mean_argmax_norm_results', ...
            'mean_half_width2_results', 'mean_argmax_norm2_results', ...
            'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
    end

    if data_randomization == 1 || data_randomization == 2
        fprintf('Reshuffling data\n');
        assert(all(size(current_tuning_function) == [N, N_SAMPLES, N_OBJECTS]));
        
        if data_randomization == 1
            % Random permutation
            pI = randperm(N_SAMPLES*N_OBJECTS);
            permuted_tuning = reshape(current_tuning_function, [N, N_SAMPLES*N_OBJECTS]);
            permuted_tuning = permuted_tuning(:,pI);
            current_tuning_function = reshape(permuted_tuning, [N, N_SAMPLES, N_OBJECTS]);
        else
            % Random permutation respecting number of samples per object
            n_samples = squeeze(sum(all(~isnan(current_tuning_function),1),2));
            assert(length(n_samples) == N_OBJECTS);
            n_valid_samples = sum(n_samples);
            pI = randperm(n_valid_samples);
            permuted_tuning = reshape(current_tuning_function, [N, N_SAMPLES*N_OBJECTS]);
            validI = all(~isnan(permuted_tuning),1);
            assert(sum(validI) == n_valid_samples);
            permuted_tuning = permuted_tuning(:,validI);
            current_tuning_function = nan(N, N_SAMPLES, N_OBJECTS);
            first = 0;
            for i=1:N_OBJECTS
                current_tuning_function(:,1:n_samples(i),i) = permuted_tuning(:,pI(first+1:first+n_samples(i)));
                first = first + n_samples(i);
            end
        end
        centers = reshape(nanmean(current_tuning_function, 2), [N, N_OBJECTS]);
    end

    % Calculate tuning in the null-space of the common space
    if Kopt == 0
        projected_tuning_function = current_tuning_function;
    else
    projected_tuning_function = current_tuning_function - reshape(Vk*(Vk'*reshape(current_tuning_function, [N, N_SAMPLES*N_OBJECTS])), size(current_tuning_function));
    end
    projected_centers = squeeze(nanmean(projected_tuning_function, 2));

    if n_neurons == 0
        features_used = 1:N; % Calculate properties once using all features
        n = N;
    else
        assert(n_neurons <= N, 'Not enough neurons to sample from');
        features_used = sample_indices(N, n_neurons, N_NEURON_SAMPLES);
        n = n_neurons;
    end
    assert(all(size(features_used) == [N_NEURON_SAMPLES, n]));
    
    for ir=1:N_NEURON_SAMPLES
        I = features_used(ir,:);
        a = spectral_dimension_results(s, ir, :);
        if all(isfinite(a(:)))
            fprintf('Skipping existing results for spectral properties...\n');
        else
            tic;
            for j=1:N_OBJECTS
                F = projected_tuning_function(I,:,j);

                % Remove missing samples
                J = all(isfinite(F),1);
                F = F(:,J);

                cF = squeeze(bsxfun(@minus, F, projected_centers(I,j)));
                n_samples = sum(J);
                minN = min(n, n_samples);
                si = svd(cF, 0);
                assert(length(si) == minN);
                sq = si.^2/n_samples;
                spectral_dimension_results(s, ir, j) = sum(sq).^2./sum(sq.^2);
                spectral_radius_results(s, ir, j) = sqrt(sum(sq.^2)) ./ norm(projected_centers(I,j)).^2;
            end
            fprintf('#%d spectral R=%1.3f D=%1.3f (took %1.1f sec)\n', ir, ...
                nanmean(mean(spectral_radius_results(s, :, :),3), 2), nanmean(mean(spectral_dimension_results(s, :, :),3), 2), toc);
            fprintf('Saving results...\n');
            save(out_name, 'AllKs', 'best_K_results', 'effective_n_results', 'theory_capacity_results', ...
                'residual_centers_results', 'centers_norm_results', ...
                'common_subspace_results', 'mean_square_corrcoef_results', 'mean_abs_corrcoef_results', ...
                'mean_square_correlations_results', 'mean_abs_correlations_results', ...
                'correlation_spectrum_results', 'normalized_correlation_spectrum_results', ...
                'original_correlations_results', 'residual_correlations_results', ...
                'effective_dimension_results', 'effective_dimension2_results', 'alphac_hat_results', ...
                'mean_half_width_results', 'mean_argmax_norm_results', ...
                'mean_half_width2_results', 'mean_argmax_norm2_results', ...
                'spectral_dimension_results', 'spectral_radius_results', '-v7.3');
        end

        if properties_method == 2
            a = mean_half_width_results(s, ir, :);
        else
            a = mean_half_width2_results(s, ir, :);
        end
        if all(isfinite(a(:)))
            fprintf('Skipping existing results for manifold properties...\n');
        else
            tic;
            if properties_method == 2
                [mean_half_width1, mean_argmax_norm1] = ...
                    check_hierarchial_capacity(current_tuning_function, I, centers, Xk, use_method);
                mean_half_width_results(s, ir, :) = mean_half_width1;
                mean_argmax_norm_results(s, ir, :) = mean_argmax_norm1;
                gaussian_dimension = mean(mean_half_width1(:).^2./mean_argmax_norm1(:));
                gaussian_radius = mean(sqrt(mean_argmax_norm1(:)));
                fprintf('#%d Rg=%1.3f Dg=%1.3f (took %1.1f sec)\n', ir, gaussian_radius, gaussian_dimension, toc);
            elseif bitand(properties_method, 4) == 4
                [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
                    check_samples_capacity(projected_tuning_function, I, projected_centers, use_method);
                mean_half_width_results(s, ir, :) = mean_half_width1;
                mean_argmax_norm_results(s, ir, :) = mean_argmax_norm1;
                mean_half_width2_results(s, ir, :) = mean_half_width2;
                mean_argmax_norm2_results(s, ir, :) = mean_argmax_norm2;
                effective_dimension_results(s, ir, :) = effective_dimension;
                effective_dimension2_results(s, ir, :) = effective_dimension2;
                alphac_hat_results(s, ir, :) = alphac_hat;
                %alphac_hat2=theory_alpha0_cached(mean_half_width2).*(1+mean_argmax_norm2);
                %Ac = mean(1./mean(1./alphac_hat2,2),1);
                Ac = mean(1./mean(1./alphac_hat,2),1);
                theory_capacity_results(s) = Ac;
                fprintf('#%d k=%d: Ac=%1.3f Rm=%1.3f Dm=%1.3f (took %1.1f sec)\n', ir, Kopt, Ac, mean(sqrt(mean_half_width2(:))), mean(effective_dimension2(:)), toc);
            else
                [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat] = ...
                    check_hierarchial_capacity(current_tuning_function, I, centers, Xk, use_method);
                mean_half_width_results(s, ir, :) = mean_half_width1;
                mean_argmax_norm_results(s, ir, :) = mean_argmax_norm1;
                mean_half_width2_results(s, ir, :) = mean_half_width2;
                mean_argmax_norm2_results(s, ir, :) = mean_argmax_norm2;
                effective_dimension_results(s, ir, :) = effective_dimension;
                effective_dimension2_results(s, ir, :) = effective_dimension2;
                alphac_hat_results(s, ir, :) = alphac_hat;
                Ac = mean(1./mean(1./alphac_hat,2),1);
                theory_capacity_results(s) = Ac;
                fprintf('#%d k=%d: Ac=%1.3f Rm=%1.3f Dm=%1.3f (took %1.1f sec)\n', ir, Kopt, Ac, mean(sqrt(mean_half_width2(:))), mean(effective_dimension2(:)), toc);
            end

            fprintf('Saving results...\n');
            save(out_name, 'AllKs', 'best_K_results', 'effective_n_results', 'theory_capacity_results', ...
                'residual_centers_results', 'centers_norm_results', ...
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
fprintf('Done. (took %1.1f sec)\n', toc(T));
end
