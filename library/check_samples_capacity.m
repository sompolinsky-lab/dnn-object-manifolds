function [mean_half_width1_results, mean_argmax_norm1_results, mean_half_width2_results, mean_argmax_norm2_results, ...
    effective_dimension_results, effective_dimension2_results, alphac_hat_results, alpha_kappa_results] = ...
    check_samples_capacity(full_tuning_function, features_used, Centers, use_method, KAPPAs, verbose, OBJECTS, D, center_norm_factor)
    % Calculate the theoretical capacity in the data sampled using the given features.
    if nargin < 3
        Centers = [];
    end
    if nargin < 4
        use_method = 0;
    end
    if nargin < 5
        KAPPAs = [];
    end
    if nargin < 6
        verbose = false;
    end
    N_KAPPAS = length(KAPPAs);

    % Verify dimensions
    [N_NEURON_SAMPLES, Nc] = size(features_used);
    assert(length(size(full_tuning_function)) == 3, 'Data must be [N_NEURONS, N_SAMPLES, N_OBJECTS]');
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(full_tuning_function);
    if nargin < 7
        OBJECTS = 1:N_OBJECTS;
    end
    if nargin < 8 || isnan(D)
        D = min(N_NEURONS, N_SAMPLES);  % Clip the PCA dimensions of each manifold to this valud
    end
    if nargin < 9
        center_norm_factor = 1;         % Scale the centers by this factor (default to 1, unchanged)
    end

    %features_used = sample_indices(N_NEURONS, Nc, N_NEURON_SAMPLES);
    %assert(all([N_NEURON_SAMPLES, Nc] == size(features_used)));

    if isempty(Centers) 
        Centers = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
    else
        assert(all(size(Centers) == [N_NEURONS, N_OBJECTS]));
    end

    N_RANDOM_PROJECTIONS = 1000;
    
    mean_half_width1_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm1_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_half_width2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    mean_argmax_norm2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    effective_dimension2_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    alphac_hat_results = nan(N_NEURON_SAMPLES, N_OBJECTS);
    alpha_kappa_results = nan(N_KAPPAS, N_NEURON_SAMPLES, N_OBJECTS);
    
    for r=1:N_NEURON_SAMPLES
        % Prepare features
        I = features_used(r, :);
        
        for i=OBJECTS
            fprintf('.');
            tuning_function = full_tuning_function(I,:,i);
            % Remove missing samples
            J = all(isfinite(tuning_function),1);
            tuning_function = tuning_function(:,J);

            % Calculate object center norm
            center = Centers(I,i);
            center_norm = sqrt(sum((center).^2, 1))*center_norm_factor;

            % Substract the center
            cF = bsxfun(@minus, tuning_function, center);
            [U, S, V] = svd(cF, 0);
            
            % Create a tuning function clipped to D dimensions
            SD = zeros(size(S)); SD(1:D,1:D) = S(1:D,1:D);
            cF_D = U*SD*V';
            tuning_function_D = bsxfun(@plus, cF_D, center*center_norm_factor);
            
            if nargout <= 2
                % Calculate only the needed manifold properties
                if use_method == 2
                    [mean_half_width1, mean_argmax_norm1] = ...
                        calc_manifold_properties3(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS);
                elseif use_method == 1
                    [mean_half_width1, mean_argmax_norm1] = ...
                        calc_manifold_properties2(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS);
                else
                    [mean_half_width1, mean_argmax_norm1] = ...
                        calc_manifold_properties(cF_D, center_norm, N_RANDOM_PROJECTIONS);
                end
                mean_half_width1_results(r,i) = mean_half_width1;
                mean_argmax_norm1_results(r,i) = mean_argmax_norm1;
            else
                % Calculate all manifold properties
                if use_method == 2
                    [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, alphac_hat2] = ...
                        calc_manifold_properties3(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS, 0);
                elseif use_method == 1
                    [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, alphac_hat2] = ...
                        calc_manifold_properties2(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS, 0);
                else
                    [mean_half_width1, mean_argmax_norm1, mean_half_width2, mean_argmax_norm2, effective_dimension, effective_dimension2, alphac_hat, alphac_hat2] = ...
                        calc_manifold_properties(cF_D, center_norm, N_RANDOM_PROJECTIONS, 0);
                end
                mean_half_width1_results(r,i) = mean_half_width1;
                mean_argmax_norm1_results(r,i) = mean_argmax_norm1;
                mean_half_width2_results(r,i) = mean_half_width2;
                mean_argmax_norm2_results(r,i) = mean_argmax_norm2;
                effective_dimension_results(r,i) = effective_dimension;
                effective_dimension2_results(r,i) = effective_dimension2;
                alphac_hat_results(r,i) = alphac_hat;
            end
            if nargout >= 7
                for k=1:N_KAPPAS
                    if use_method == 2
                        [~, ~, ~, ~, ~, ~, alphac_hat] = calc_manifold_properties3(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS, KAPPAs(k));
                    elseif use_method == 1
                        [~, ~, ~, ~, ~, ~, alphac_hat] = calc_manifold_properties2(tuning_function_D, center_norm, N_RANDOM_PROJECTIONS, KAPPAs(k));
                    else
                        [~, ~, ~, ~, ~, ~, alphac_hat] = calc_manifold_properties(cF_D, center_norm, N_RANDOM_PROJECTIONS, KAPPAs(k));
                    end
                    alpha_kappa_results(k, r, i) = alphac_hat;
                end
            end
        end
        fprintf('\n');
    end

    if verbose
        alpha_hat = mean(1./mean(1./alphac_hat_results, 2), 1);
        fprintf(' N=%d alpha_hat=%1.2f (D=%1.2f Rhw=%1.2f)\n', Nc, alpha_hat, mean(effective_dimension_results(:)), mean(mean_half_width1_results(:)));
    end
end
